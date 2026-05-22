"""TinyGNN: the small network applied recursively over the K slot nodes.

Because K is tiny (7), the "graph" is dense all-pairs, so message passing is just
multi-head self-attention over the slots followed by a per-slot FFN. This is the
single reusable block of the TRM recursion.

It consumes the three TRM streams and returns updated answer/latent streams:

    (x, y, z) -> (y_next, z_next)         all (B, K, slot_dim)

    x : perceptual evidence (initial slots), frozen across the recursion ("question")
    y : current slot answer, refined each step
    z : per-slot latent reasoning state, carried across steps

The streams are combined by summation (after per-stream LayerNorm), passed through
``n_layers`` pre-norm transformer blocks, and the shared hidden state drives residual
updates of z then y (TRM-style: latent updated, then answer read out from it).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Block(nn.Module):
    """Pre-norm multi-head self-attention + FFN over the K slot tokens."""

    def __init__(self, dim: int, n_heads: int = 4, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class TinyGNN(nn.Module):
    def __init__(
        self,
        slot_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        n_latent_steps: int = 1,
        combine: str = "concat",
    ) -> None:
        super().__init__()
        assert combine in {"sum", "concat"}, combine
        self.n_latent_steps = n_latent_steps
        self.combine = combine

        # Per-stream input norms before combining (keeps scales comparable).
        self.norm_x = nn.LayerNorm(slot_dim)
        self.norm_y = nn.LayerNorm(slot_dim)
        self.norm_z = nn.LayerNorm(slot_dim)
        # "concat": a learned projection of [x, y, z] lets the network gate/route the
        # (re-bound, in coupled mode) perceptual evidence against the reasoning state,
        # rather than forcing them into one additive space ("sum").
        if combine == "concat":
            self.combine_proj = nn.Linear(3 * slot_dim, slot_dim)

        self.blocks = nn.ModuleList(
            [_Block(slot_dim, n_heads=n_heads) for _ in range(n_layers)]
        )

        # Residual read-out heads for the two streams.
        self.to_z = nn.Linear(slot_dim, slot_dim)
        self.to_y = nn.Linear(slot_dim, slot_dim)
        nn.init.zeros_(self.to_z.weight)
        nn.init.zeros_(self.to_z.bias)
        nn.init.zeros_(self.to_y.weight)
        nn.init.zeros_(self.to_y.bias)

    def _core(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        nx, ny, nz = self.norm_x(x), self.norm_y(y), self.norm_z(z)
        if self.combine == "concat":
            h = self.combine_proj(torch.cat([nx, ny, nz], dim=-1))
        else:
            h = nx + ny + nz
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Update the latent z one or more times, then read the answer y off it.
        for _ in range(self.n_latent_steps):
            h = self._core(x, y, z)
            z = z + self.to_z(h)
        h = self._core(x, y, z)
        y = y + self.to_y(h)
        return y, z
