"""Training loop: bf16 mixed precision, deep supervision, gradient clipping, ACT.

Launch (after data is downloaded):
    uv run python -m dino_trm.train                       # baseline, VOC defaults
    uv run python -m dino_trm.train mode=trm
    uv run python -m dino_trm.train mode=coupled loss.entropy_weight=0.01
    uv run python -m dino_trm.train log.wandb=false        # quick local run

The frozen DINOv3 backbone runs under no_grad inside the model, so the optimizer only
sees the ~3.8M trainable params (slot attention + decoder + recursion).
"""

from __future__ import annotations

import json
import math
import os
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .data.pascal_voc import build_loader
from .data.voc_imageonly import build_imageonly_loader
from .eval_protocol import evaluate_protocol
from .losses import total_training_loss
from .models.full_model import DinoSlotModel
from .utils.logging import Logger
from .utils.viz import recursion_evolution_figure


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@hydra.main(version_base=None, config_path="../../configs", config_name="pascal_voc")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(OmegaConf.to_yaml(cfg))

    model = DinoSlotModel(mode=cfg.mode, **cfg.model).to(device)
    grid_size = model.backbone.grid_size
    print(f"mode={cfg.mode} grid={grid_size} num_patches={model.num_patches}")

    dataset = cfg.data.get("dataset", "voc")
    if dataset == "coco":
        from .data.coco import build_coco_loader
        coco_root = cfg.data.get("coco_root", "data/coco")
        train_loader = build_coco_loader(
            split="train",
            batch_size=cfg.data.batch_size,
            image_size=cfg.data.image_size,
            num_workers=cfg.data.num_workers,
            root=coco_root,
        )
        val_loader = build_coco_loader(
            split="val",
            batch_size=cfg.data.batch_size,
            image_size=cfg.data.image_size,
            num_workers=cfg.data.num_workers,
            root=coco_root,
        )
        print(f"train set: COCO multi-object subset, {len(train_loader.dataset)} images; "
              f"val {len(val_loader.dataset)} images")
    else:
        if cfg.data.get("large_train", False):
            train_loader = build_imageonly_loader(
                batch_size=cfg.data.batch_size,
                image_size=cfg.data.image_size,
                num_workers=cfg.data.num_workers,
                cache_dir=cfg.data.cache_dir,
            )
            print(f"train set: VOC image-only (leakage-free), {len(train_loader.dataset)} images")
        else:
            train_loader = build_loader(
                split="train",
                batch_size=cfg.data.batch_size,
                image_size=cfg.data.image_size,
                grid_size=grid_size,
                num_workers=cfg.data.num_workers,
                cache_dir=cfg.data.cache_dir,
            )
        val_loader = build_loader(
            split="val",
            batch_size=cfg.data.batch_size,
            image_size=cfg.data.image_size,
            grid_size=grid_size,
            num_workers=cfg.data.num_workers,
            cache_dir=cfg.data.cache_dir,
            full_mask=True,  # published-protocol eval needs image-resolution masks
        )

    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    print(f"trainable params: {n_trainable/1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    steps_per_epoch = len(train_loader) // cfg.optim.grad_accum
    total_steps = steps_per_epoch * cfg.optim.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, cfg.optim.warmup_steps, total_steps)
    )

    logger = Logger(
        enabled=cfg.log.wandb,
        project=cfg.log.project,
        run_name=cfg.log.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.log.mode,
    )

    # Fixed validation batch for consistent recursion-evolution visualisations.
    viz_batch = next(iter(val_loader))
    os.makedirs(cfg.log.ckpt_dir, exist_ok=True)

    # Local results dir (independent of wandb): metrics history + figures.
    # Namespace by dataset so COCO runs don't clobber the VOC figures/metrics
    # (and vice versa); VOC keeps the original flat path for back-compat.
    run_tag = cfg.mode if dataset == "voc" else os.path.join(dataset, cfg.mode)
    results_dir = os.path.join("results", run_tag)
    os.makedirs(results_dir, exist_ok=True)
    metrics_history: list[dict] = []

    global_step = 0
    for epoch in range(cfg.optim.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_loader):
            px = batch["pixel_values"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(px)
            loss, logs = total_training_loss(
                out,
                intermediate_weight=cfg.loss.intermediate_weight,
                final_weight=cfg.loss.final_weight,
                entropy_weight=cfg.loss.entropy_weight,
                act_weight=cfg.loss.act_weight,
                balance_weight=cfg.loss.balance_weight,
                query_ortho_weight=cfg.loss.query_ortho_weight,
                query_ortho_margin=cfg.loss.query_ortho_margin,
            )
            (loss / cfg.optim.grad_accum).backward()

            if (it + 1) % cfg.optim.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 50 == 0:
                    logs["lr"] = sched.get_last_lr()[0]
                    logs["grad_norm"] = float(grad_norm)
                    logs["epoch"] = epoch
                    logger.log(logs, step=global_step)

                if global_step % cfg.log.viz_every == 0:
                    fig_path = os.path.join(results_dir, f"masks_step{global_step}.png")
                    _log_viz(model, viz_batch, device, grid_size, logger, global_step, fig_path)

        if (epoch + 1) % cfg.log.eval_every_epochs == 0:
            metrics = evaluate_protocol(model, val_loader, device, max_batches=cfg.eval.max_batches)
            log_metrics = {f"val/{k}": v for k, v in metrics.items() if not isinstance(v, list)}
            for t, v in enumerate(metrics.get("fg_ari_per_step", [])):
                log_metrics[f"val/fg_ari_step_{t}"] = v
            log_metrics["epoch"] = epoch
            logger.log(log_metrics, step=global_step)
            print(f"[epoch {epoch}] mBO_i={metrics['mbo_i']:.4f} mBO_c={metrics['mbo_c']:.4f} "
                  f"FG-ARI={metrics['fg_ari']:.4f}")

            metrics_history.append({"epoch": epoch, "step": global_step, **metrics})
            with open(os.path.join(results_dir, "metrics.json"), "w") as f:
                json.dump({"mode": cfg.mode, "history": metrics_history}, f, indent=2)
            torch.save(
                {"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg), "epoch": epoch},
                os.path.join(cfg.log.ckpt_dir, f"{cfg.mode}_epoch{epoch}.pt"),
            )

    # Persist the final recursion-evolution figure for the qualitative deliverable.
    _log_viz(model, viz_batch, device, grid_size, logger, global_step,
             os.path.join(results_dir, "masks_final.png"))
    logger.finish()


@torch.no_grad()
def _log_viz(model, batch, device, grid_size, logger, step, save_path=None) -> None:
    model.eval()
    px = batch["pixel_values"][:1].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(px)
    masks_per_step = [m[0] for m in out["masks_list"]]
    fig = recursion_evolution_figure(px[0], masks_per_step, grid_size)
    logger.log_image("val/recursion_masks", fig, step=step)
    if save_path is not None:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    model.train()


if __name__ == "__main__":
    main()
