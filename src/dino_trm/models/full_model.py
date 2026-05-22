"""Wires backbone -> input projection -> slot attention -> (recursion) -> decoder.

Three modes select the architecture phase:

  * "baseline" (Phase 1): slot attention once, decode, reconstruct features.
  * "trm"      (Phase 2): slot attention once -> TRM recursion refines slots; decode
                          every step for deep supervision. No feedback into binding.
  * "coupled"  (Phase 3): at every recursion step slot attention re-binds conditioned
                          on the current latent z (top-down feedback), then TinyGNN
                          reasons. The novel coupled loop.

Recursive modes return per-step lists so the trainer can apply deep supervision and
plot metric-vs-recursion-depth. Baseline returns length-1 lists for a uniform API.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import DINOV3_S, FrozenBackbone
from .decoder import SpatialBroadcastDecoder
from .slot_attention import SlotAttention
from .trm_module import TRMReasoner


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
        n_steps: int = 8,
        gnn_layers: int = 2,
        gnn_heads: int = 4,
        n_latent_steps: int = 1,
        gnn_combine: str = "concat",
        image_size: int = 336,
        target_norm: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        assert mode in {"baseline", "trm", "coupled"}, mode
        self.mode = mode
        self.num_slots = num_slots
        self.target_norm = target_norm

        self.backbone = FrozenBackbone(model_id=model_id, dtype=dtype, image_size=image_size)
        feat_dim = self.backbone.feature_dim
        self.num_patches = self.backbone.num_patches
        self.grid_size = self.backbone.grid_size

        self.input_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, slot_dim),
        )
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=slot_dim,
            n_iters=slot_iters,
        )
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=slot_dim,
            feature_dim=feat_dim,
            num_patches=self.num_patches,
            hidden_dim=decoder_hidden,
            n_layers=decoder_layers,
        )
        self.reasoner = (
            TRMReasoner(
                slot_dim=slot_dim,
                n_steps=n_steps,
                gnn_layers=gnn_layers,
                gnn_heads=gnn_heads,
                n_latent_steps=n_latent_steps,
                gnn_combine=gnn_combine,
            )
            if mode in {"trm", "coupled"}
            else None
        )

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
        slots0, attn0, q0 = self.slot_attention(proj, return_queries=True)  # initial binding
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

        rebind = None
        if self.mode == "coupled":
            def rebind(z, _proj=proj):
                slots, _, q = self.slot_attention(_proj, z_prev=z, return_queries=True)
                queries.append(q)  # regularise the z-conditioned queries each step
                return slots

        out = self.reasoner(slots0, rebind=rebind)

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
