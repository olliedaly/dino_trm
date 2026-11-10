"""Per-image scores for the slot-attention-loop arms (docs/sa_loop_plan.md).

For each run and each pass count n, every validation image goes through
backbone -> projection -> slot attention run for n passes -> decoder, and two readouts
are scored:
  map{n}   argmax over slots of slot attention's last attention map
  dec{n}   argmax over slots of the decoder's masks
with FG-ARI, mBO_i and mBO_c (the functions of dino_trm.eval_protocol, at image
resolution), ARI counting background (all pixels, background as one ground-truth
segment) and the reconstruction error. Every n gets the same initial slot noise.

Only for models without a refinement module (mode=baseline or mode=sa_rec). The run's
final checkpoint is required: checkpoints/coco/{run}_epoch{epochs-1}.pt.

Writes results/sa_loop/per_image/{run}.npz: "{readout}{n}|{metric}" -> (N,) float32,
"err{n}" -> (N,) reconstruction error, "meta" -> a JSON string.

    source scripts/sa_loop/env.sh && uv run python scripts/sa_loop/eval_arms.py \
        --runs baseline_pm,baseline_pm_seed1 [--passes 3,8] [--eval-seeds 0] [--force]
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from dino_trm.data.coco import build_coco_loader
from dino_trm.eval_protocol import _best_overlap, _fg_ari, _upsample_pred
from dino_trm.models.full_model import DinoSlotModel

CKPT_DIR = "checkpoints/coco"
OUT_DIR = "results/sa_loop/per_image"
NAN = float("nan")


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load(run: str):
    """The run's final checkpoint. Fails if training did not finish."""
    first = os.path.join(CKPT_DIR, f"{run}_epoch0.pt")
    if not os.path.exists(first):
        raise SystemExit(f"{run}: no checkpoint {first}")
    epochs = torch.load(first, map_location="cpu", weights_only=False)["cfg"]["optim"]["epochs"]
    path = os.path.join(CKPT_DIR, f"{run}_epoch{epochs - 1}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"{run}: training did not finish, {path} is missing")
    c = torch.load(path, map_location="cuda", weights_only=False)
    cfg = c["cfg"]
    assert c["epoch"] == epochs - 1, (run, c["epoch"])
    assert cfg["mode"] in ("baseline", "sa_rec"), f"{run}: mode {cfg['mode']} has a module; not supported here"
    assert cfg["data"].get("dataset") == "coco", run
    model = DinoSlotModel(mode=cfg["mode"], **cfg["model"]).cuda()
    model.load_state_dict(c["model"])
    return model.eval(), cfg, path


def score(pred: np.ndarray, inst: np.ndarray, sem: np.ndarray) -> dict:
    """The eval protocol's three scores, plus ARI over all pixels."""
    cls_ids = [c for c in np.unique(sem) if c != 0]
    inst_ids = [i for i in np.unique(inst) if i != 0]
    f, mi, mc = _fg_ari(pred, inst), _best_overlap(pred, inst_ids, inst), _best_overlap(pred, cls_ids, sem)
    return {"fg_ari": NAN if f is None else f, "mbo_i": NAN if mi is None else mi,
            "mbo_c": NAN if mc is None else mc,
            "ari_bg": float(adjusted_rand_score(inst.ravel(), pred.ravel()))}


@torch.no_grad()
def run_one(run: str, passes: list[int], eval_seed: int) -> dict:
    model, cfg, path = load(run)
    size, grid = model.backbone.image_size, model.backbone.grid_size
    assert size == cfg["data"]["image_size"], (run, size, cfg["data"]["image_size"])
    loader = build_coco_loader(split="val", batch_size=16, image_size=size, num_workers=4,
                               shuffle=False, return_masks=True)
    trained = cfg["model"]["slot_iters"]
    keep: dict[str, list] = {}
    set_seed(eval_seed)
    for batch in loader:
        px = batch["pixel_values"].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = model.backbone(px).float()
            if model.target_norm:
                feats = (feats - feats.mean(1, keepdim=True)) / (feats.std(1, keepdim=True) + 1e-6)
            proj = model.input_proj(feats)
            rng = torch.cuda.get_rng_state()
            outs = {}
            for n in passes:
                torch.cuda.set_rng_state(rng)  # the same initial slot noise for every n
                model.slot_attention.n_iters = n
                slots, attn = model.slot_attention(proj)
                recon, masks = model.decoder(slots)
                err = (recon.float() - feats).pow(2).mean(dim=(1, 2))
                outs[n] = (attn.float(), masks.float(), err.cpu().numpy())
        model.slot_attention.n_iters = trained
        for b in range(px.shape[0]):
            sem = batch["label_full"][b].numpy().copy()
            sem[sem == 255] = 0  # void -> background, as in the eval protocol
            inst = batch["inst_full"][b].numpy()
            for n, (attn, masks, err) in outs.items():
                keep.setdefault(f"err{n}", []).append(err[b])
                for name, m in (("map", attn), ("dec", masks)):
                    pred = _upsample_pred(m[b].view(-1, grid, grid), size)
                    for k, v in score(pred, inst, sem).items():
                        keep.setdefault(f"{name}{n}|{k}", []).append(v)
    out = {k: np.asarray(v, np.float32) for k, v in keep.items()}
    out["_meta"] = dict(run=run, ckpt=path, mode=cfg["mode"], trained_passes=trained, seed=cfg["seed"],
                        image_size=size, sa_detach=cfg["model"].get("sa_detach", False),
                        intermediate_weight=cfg["loss"]["intermediate_weight"],
                        decoder_hidden=cfg["model"]["decoder_hidden"],
                        params_M=round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 4))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run names")
    ap.add_argument("--passes", default="3,8")
    ap.add_argument("--eval-seeds", default="0", help="comma-separated; scores are averaged per image")
    ap.add_argument("--force", action="store_true", help="re-score runs that already have a file")
    a = ap.parse_args()
    passes = [int(x) for x in a.passes.split(",")]
    seeds = [int(x) for x in a.eval_seeds.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)
    for run in a.runs.split(","):
        dst = os.path.join(OUT_DIR, f"{run}.npz")
        if os.path.exists(dst) and not a.force:
            print(f"{run}: {dst} exists, skipped (use --force to redo)", flush=True)
            continue
        per_seed = [run_one(run, passes, s) for s in seeds]
        meta = per_seed[0].pop("_meta")
        for d in per_seed[1:]:
            d.pop("_meta")
        meta.update(passes=passes, eval_seeds=seeds)
        out = {k: np.mean([d[k] for d in per_seed], 0).astype(np.float32) for k in per_seed[0]}
        np.savez_compressed(dst, meta=json.dumps(meta), **out)
        n = meta["trained_passes"]
        if n not in passes:
            n = passes[0]
        show = "  ".join(f"{k} {100 * np.nanmean(out[f'map{n}|{k}']):.2f}" for k in ("fg_ari", "mbo_i", "mbo_c", "ari_bg"))
        print(f"{run}: {len(out[f'err{n}'])} images, {meta['params_M']}M params, trained with {meta['trained_passes']} "
              f"passes | map at {n} passes: {show} | error x1e3 {1e3 * out[f'err{n}'].mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
