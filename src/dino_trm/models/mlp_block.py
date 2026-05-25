"""MLPBlockReasoner: non-recursive feedforward control for the TRM recursion.

Drop-in replacement for ``TRMReasoner`` in the architectural slot between the
initial slot attention and the decoder. Applies ``n_layers`` of the *same*
transformer ``_Block`` used by TinyGNN (self-attn over the K slots + optional
slot→patch cross-attn + FFN) **once**, with no recursion and no (x, y, z) latent
streams.

Why: TRM/coupled win ~+2.3 mBO_i / +4 FG-ARI over the no-reasoner baseline on the
COCO multi-object subset, but they also add ~2.5 M trainable params via TinyGNN +
halt-head + z0. This module is the parameter control: same patch-grounded
self/cross-attn capacity in the same place in the pipeline, but feedforward. If
``mlp_block`` lands near baseline while ``trm``/``coupled`` stay at +2.3, the gain
is from the recursion (the scientific claim) and not from raw capacity.

Default ``n_layers=3`` makes the trainable-param count slightly *exceed* the
recursive variants (~3.2 M vs ~2.5 M in this module's slot), so the control is
conservative — it cannot be dismissed as "fewer params".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .tiny_gnn import _Block


class MLPBlockReasoner(nn.Module):
    def __init__(
        self,
        slot_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.cross_attn = cross_attn
        self.blocks = nn.ModuleList(
            [_Block(slot_dim, n_heads=n_heads, cross_attn=cross_attn) for _ in range(n_layers)]
        )
        # Residual read-out head; zero-init so the block starts as the identity and the
        # initial slots are a strict fixed point at init (matches TinyGNN's to_y / to_z
        # convention — keeps training stable from step 0).
        self.out_norm = nn.LayerNorm(slot_dim)
        self.to_y = nn.Linear(slot_dim, slot_dim)
        nn.init.zeros_(self.to_y.weight)
        nn.init.zeros_(self.to_y.bias)

    def forward(
        self,
        slots: torch.Tensor,                       # (B, K, slot_dim) initial slots
        patches: torch.Tensor | None = None,       # (B, N, slot_dim) patch features
    ) -> torch.Tensor:
        """slots → slots + delta. Returns refined slots (B, K, slot_dim)."""
        h = slots
        for blk in self.blocks:
            h = blk(h, patches)
        return slots + self.to_y(self.out_norm(h))
