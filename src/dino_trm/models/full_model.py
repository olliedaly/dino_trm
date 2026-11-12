"""Wires backbone -> input projection -> slot attention -> (recursion) -> decoder.

Five modes select the architecture phase:

  * "baseline"  (Phase 1): slot attention once, decode, reconstruct features.
  * "trm"       (Phase 2): slot attention once -> TRM recursion refines slots; decode
                           every step for deep supervision. No feedback into binding.
  * "coupled"   (Phase 3): at every recursion step slot attention re-binds conditioned
                           on the current latent z (top-down feedback), then TinyGNN
                           reasons. The novel coupled loop.
  * "mlp_block" (Control): slot attention once -> non-recursive transformer stack over
                           slots (same _Block as TinyGNN, optionally with cross-attn) ->
                           decode. Parameter-matched to trm/coupled; isolates the
                           contribution of the recursion itself from raw capacity.
  * "sa_rec"    (Slot-attention recursion): no module; slot attention's own iterations
                           are the recursion. The slots after every iteration are decoded
                           (deep supervision, full backprop through all iterations), so
                           slot attention itself must do the grouping. Set the depth
                           with ``slot_iters``. Same parameters as the baseline.
                           ``sa_detach=True`` detaches the slots between iterations
                           (1-step gradients): full backprop through 8 iterations
                           stopped slot attention converging (sa_rec run, COCO).

Recursive modes return per-step lists so the trainer can apply deep supervision and
plot metric-vs-recursion-depth. Baseline / mlp_block return length-1 lists for a
uniform API.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import DINOV3_S, FrozenBackbone
from .decoder import SpatialBroadcastDecoder
from .mlp_block import MLPBlockReasoner
from .slot_attention import SlotAttention
from .trm_module import TRMReasoner


# Deep-supervision weights of configs/base.yaml (loss.intermediate_weight, loss.final_weight).
SA_PASS_WEIGHTS = (0.25, 1.0)


def sample_passes(n_passes: int, batch: int) -> torch.Tensor:
    """One slot-attention pass per image, for mode "sa_rec" with ``sa_sample_pass``.

    Each image's pass is drawn with probability proportional to its deep-supervision
    weight (0.25 for an intermediate pass, 1.0 for the final one). A loss on each image's
    own pass has the same expected gradient direction as the weighted loss on every pass
    of every image, at the cost of one decode per image instead of ``n_passes``. Every
    step still mixes all the passes, which a single pass per step would not.
    Returns (batch,) long.
    """
    w = torch.tensor([SA_PASS_WEIGHTS[0]] * (n_passes - 1) + [SA_PASS_WEIGHTS[1]])
    return torch.multinomial(w, batch, replacement=True)


class DinoSlotModel(nn.Module):
    def __init__(
        self,
        mode: str = "baseline",
        model_id: str = DINOV3_S,
        num_slots: int = 7,
        slot_dim: int = 256,
        slot_iters: int = 3,
        decoder_hidden: int = 1024,
        decoder_layers: int = 4,
        sa_mlp_hidden: int | None = None,  # DINOSAUR E.1: slot-attention residual-MLP hidden
                                           # = 4*D_slots. None = keep original 512 (checkpoint compat).
        pre_sa_mlp: bool = False,          # DINOSAUR E.1: two-layer pre-slot-attention MLP.
                                           # False = original single Linear (checkpoint compat).
        n_steps: int = 8,
        gnn_layers: int = 2,
        gnn_heads: int = 4,
        n_latent_steps: int = 1,
        gnn_combine: str = "concat",
        gnn_cross_attn: bool = False,
        mlp_block_layers: int = 3,   # depth of the non-recursive control reasoner
        sa_detach: bool = False,     # detach slots between slot-attention iterations (1-step grads)
        sa_sample_pass: bool = False,  # sa_rec, training: decode each image at one sampled pass (sample_passes)
        trm_detach: bool = False,    # trm/coupled: detach y, z between steps (1-step grads)
        trm_untie: bool = False,     # trm/coupled: a separate block per step (no weight sharing)
        gnn_mlp_ratio: int = 4,      # trm/coupled: FFN width of the module's blocks (x slot_dim)
        loc_bias: float = 0.0,       # decoder locality prior (0 = off); see SpatialBroadcastDecoder
        image_size: int = 336,
        target_norm: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        assert mode in {"baseline", "trm", "coupled", "mlp_block", "sa_rec"}, mode
        self.mode = mode
        self.num_slots = num_slots
        self.target_norm = target_norm
        self.sa_detach = sa_detach
        self.sa_sample_pass = sa_sample_pass

        self.backbone = FrozenBackbone(model_id=model_id, dtype=dtype, image_size=image_size)
        feat_dim = self.backbone.feature_dim
        self.num_patches = self.backbone.num_patches
        self.grid_size = self.backbone.grid_size

        if pre_sa_mlp:
            # DINOSAUR E.1: "After applying a layer norm, the set of N input features
            # is transformed to slot dimensionality D_slots by a two-layer MLP with
            # hidden size equal to the feature size D_feat, followed by another layer
            # norm." Default (False) keeps the original single-Linear projection so
            # trained checkpoints load unchanged.
            self.input_proj = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, slot_dim),
                nn.LayerNorm(slot_dim),
            )
        else:
            self.input_proj = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, slot_dim),
            )
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=slot_dim,
            n_iters=slot_iters,
            mlp_hidden=sa_mlp_hidden if sa_mlp_hidden is not None else 512,
        )
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=slot_dim,
            feature_dim=feat_dim,
            num_patches=self.num_patches,
            hidden_dim=decoder_hidden,
            n_layers=decoder_layers,
            loc_bias=loc_bias,
        )
        if mode in {"trm", "coupled"}:
            self.reasoner = TRMReasoner(
                slot_dim=slot_dim,
                n_steps=n_steps,
                gnn_layers=gnn_layers,
                gnn_heads=gnn_heads,
                n_latent_steps=n_latent_steps,
                gnn_combine=gnn_combine,
                gnn_cross_attn=gnn_cross_attn,
                detach_steps=trm_detach,
                untie_steps=trm_untie,
                gnn_mlp_ratio=gnn_mlp_ratio,
            )
        elif mode == "mlp_block":
            self.reasoner = MLPBlockReasoner(
                slot_dim=slot_dim,
                n_layers=mlp_block_layers,
                n_heads=gnn_heads,
                cross_attn=gnn_cross_attn,
            )
        else:
            self.reasoner = None

    def forward(self, pixel_values: torch.Tensor) -> dict:
        feats = self.backbone(pixel_values).float()   # (B, N, feat_dim)
        if self.target_norm:
            # Standardise each feature channel across patches (per image): keeps the
            # reconstruction target's scale uniform so no high-norm dim dominates the
            # MSE (DINOSAUR-style target normalisation). Used for both encoder + target.
            mean = feats.mean(dim=1, keepdim=True)
            std = feats.std(dim=1, keepdim=True)
            feats = (feats - mean) / (std + 1e-6)
        target = feats
        proj = self.input_proj(target)                # (B, N, slot_dim)

        if self.mode == "sa_rec":
            # Decode the slots after every slot-attention iteration (deep supervision). With
            # sa_sample_pass, in training, decode each image at one sampled iteration instead.
            slots_all, attn_all, q = self.slot_attention(
                proj, return_all=True, detach_iters=self.sa_detach)
            recons, masks_list = [], []
            if self.sa_sample_pass and self.training:
                picked = sample_passes(len(slots_all), proj.shape[0]).to(proj.device)
                chosen = torch.stack(slots_all)[picked, torch.arange(proj.shape[0], device=proj.device)]
                recon, masks = self.decoder(chosen)
                recons.append(recon)
                masks_list.append(masks)
                picked = picked.tolist()
            else:
                picked = list(range(len(slots_all)))
                for s in slots_all:
                    recon, masks = self.decoder(s)
                    recons.append(recon)
                    masks_list.append(masks)
            return {
                "target": target,
                "recons": recons,          # n_iters tensors (B, N, feat_dim)
                "masks_list": masks_list,  # n_iters tensors (B, K, N)
                "slots_list": slots_all,
                "halts": [],
                "queries": [q],
                "attn": attn_all[-1],
                "attn_list": attn_all,     # n_iters tensors (B, K, N)
                "passes": picked,          # the iterations decoded (with sa_sample_pass: one per image)
            }

        # sa_detach also applies here (off by default): mode=baseline with slot_iters=8 and
        # sa_detach=true trains slot attention through its last update only.
        slots0, attn0, q0 = self.slot_attention(
            proj, return_queries=True, detach_iters=self.sa_detach)  # initial binding
        queries = [q0]  # slot queries collected for the orthogonality loss

        if self.mode == "baseline":
            recon, masks = self.decoder(slots0)
            return {
                "target": target,
                "recons": [recon],
                "masks_list": [masks],
                "slots_list": [slots0],
                "halts": [],
                "queries": queries,
                "attn": attn0,
            }

        if self.mode == "mlp_block":
            # Param-matched non-recursive control: one pass through the feedforward
            # block, then decode once. Patches are forwarded so the cross-attn path
            # (when enabled) matches what trm/coupled get every recursion step.
            slots1 = self.reasoner(slots0, patches=proj)
            recon, masks = self.decoder(slots1)
            return {
                "target": target,
                "recons": [recon],
                "masks_list": [masks],
                "slots_list": [slots1],
                "halts": [],
                "queries": queries,
                "attn": attn0,
            }

        rebind = None
        if self.mode == "coupled":
            def rebind(z, _proj=proj):
                slots, _, q = self.slot_attention(_proj, z_prev=z, return_queries=True)
                queries.append(q)  # regularise the z-conditioned queries each step
                return slots

        out = self.reasoner(slots0, patches=proj, rebind=rebind)

        recons, masks_list = [], []
        for y in out["y"]:
            recon, masks = self.decoder(y)
            recons.append(recon)
            masks_list.append(masks)

        return {
            "target": target,
            "recons": recons,         # T tensors (B, N, feat_dim)
            "masks_list": masks_list,  # T tensors (B, K, N)
            "slots_list": out["y"],
            "halts": out["halt"],      # T tensors (B, 1), fp32
            "queries": queries,        # list of (B, K, slot_dim) binding queries
            "attn": attn0,
        }
