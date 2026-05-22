"""TRM recursive reasoning over the slot graph.

Runs the TinyGNN block for ``n_steps`` outer steps with **full backprop through the
entire recursion** (no 1-step gradient approximation — TRM showed that matters). At
each step it emits the refined answer ``y_t`` (for deep supervision) and an ACT-style
halting logit ``Q(z_t)`` computed in fp32 (bf16 over a deep recursion can NaN the
halting sigmoid).

Phase 2 (mode "trm"): slot attention runs once outside; ``x`` is held fixed and the
recursion refines ``y``.

Phase 3 (mode "coupled"): pass ``rebind`` — a callable ``z -> slots`` that re-runs
slot attention conditioned on the current latent. The freshly bound slots become the
step's perceptual evidence, so binding and reasoning interleave.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .tiny_gnn import TinyGNN


class TRMReasoner(nn.Module):
    def __init__(
        self,
        slot_dim: int = 256,
        n_steps: int = 8,
        gnn_layers: int = 2,
        gnn_heads: int = 4,
        n_latent_steps: int = 1,
        gnn_combine: str = "concat",
        gnn_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.slot_dim = slot_dim
        self.tiny_gnn = TinyGNN(
            slot_dim=slot_dim,
            n_layers=gnn_layers,
            n_heads=gnn_heads,
            n_latent_steps=n_latent_steps,
            combine=gnn_combine,
            cross_attn=gnn_cross_attn,
        )
        # Learned latent initialisation.
        self.z0 = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # ACT halting head: per-slot score -> mean over slots -> scalar logit.
        self.halt_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, 1),
        )

    def init_latent(self, x: torch.Tensor) -> torch.Tensor:
        b, k, _ = x.shape
        return self.z0.expand(b, k, self.slot_dim).to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,                                  # (B, K, slot_dim) initial slots
        patches: torch.Tensor | None = None,              # (B, N, slot_dim) patch features
        rebind: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> dict:
        """Returns per-step lists: 'y' (refined slots), 'z' (latents), 'halt' (logits).

        ``patches`` is forwarded to the TinyGNN every step so the slots can re-read the
        image via cross-attention (no-op unless the GNN was built with cross_attn).
        """
        y = x
        z = self.init_latent(x)

        ys: list[torch.Tensor] = []
        zs: list[torch.Tensor] = []
        halts: list[torch.Tensor] = []

        for _ in range(self.n_steps):
            x_t = rebind(z) if rebind is not None else x  # Phase 3 re-binding
            if rebind is not None:
                # Fresh binding also reseeds the answer stream for this step.
                y = x_t if not ys else y
            y, z = self.tiny_gnn(x_t, y, z, patches)
            # Halting score in fp32 for numerical stability over the recursion.
            halt = self.halt_head(z.float()).mean(dim=1)  # (B, 1)
            ys.append(y)
            zs.append(z)
            halts.append(halt)

        return {"y": ys, "z": zs, "halt": halts}
