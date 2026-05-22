"""Vanilla Slot Attention (Locatello et al. 2020), with an optional z-conditioning
hook used in Phase 3.

Standard module: K slots compete to explain N input features over ``n_iters``
iterations of attention + GRU update. The Locatello numerical-stability fix
(``eps`` in the attention-weight normaliser) and the GRU update are both used.

Phase-3 hook: ``forward`` accepts an optional ``z_prev`` of shape (B, K, slot_dim).
When given, slot queries are initialised as ``slot_init + W_z . z_prev`` instead of
being sampled from the learned Gaussian (mu, sigma). When ``z_prev`` is None the
module behaves exactly like standard Slot Attention, so Phases 1-2 use it untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    def __init__(
        self,
        num_slots: int = 7,
        slot_dim: int = 256,
        input_dim: int = 256,
        n_iters: int = 3,
        mlp_hidden: int = 512,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        # Learned slot initialisation (sampled per-forward unless z-conditioned).
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_log_sigma)

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, slot_dim),
        )

        # Phase-3 conditioning projection (latent z -> slot-query offset).
        self.z_to_slot = nn.Linear(slot_dim, slot_dim)
        nn.init.zeros_(self.z_to_slot.weight)
        nn.init.zeros_(self.z_to_slot.bias)

    def _init_slots(self, batch: int, device, dtype, z_prev: torch.Tensor | None) -> torch.Tensor:
        mu = self.slots_mu.expand(batch, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        if z_prev is not None:
            slots = slots + self.z_to_slot(z_prev)
        return slots.to(dtype)

    def forward(
        self,
        inputs: torch.Tensor,            # (B, N, input_dim)
        z_prev: torch.Tensor | None = None,  # (B, K, slot_dim) or None
        return_queries: bool = False,
    ):
        """Returns (slots (B, K, slot_dim), attn (B, K, N)).

        ``attn`` are the normalised per-slot attention masks over inputs from the
        final iteration (rows sum to 1 over N), useful for reconstruction-free
        segmentation metrics and visualisation. With ``return_queries=True`` also
        returns the final-iteration slot queries ``q`` (B, K, slot_dim) — the vectors
        regularised by the query-orthogonality loss to prevent collapse.
        """
        b, n, _ = inputs.shape
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        slots = self._init_slots(b, inputs.device, inputs.dtype, z_prev)

        attn = None
        q = None
        for _ in range(self.n_iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))

            # Attention logits, then a competitive softmax over the SLOTS axis.
            # The softmax MUST be in fp32: in bf16 it underflows and slots randomly
            # "die" (get zero attention -> zero gradient). Same for the weighted-mean
            # normalisation and the GRU update below.
            dots = torch.einsum("bkd,bnd->bkn", q, k) * self.scale  # (B, K, N)
            attn = dots.float().softmax(dim=1)  # fp32 competition across slots

            # Weighted mean over inputs, normalised per slot (Locatello eps fix), fp32.
            attn_wm = attn + self.eps
            attn_wm = attn_wm / attn_wm.sum(dim=-1, keepdim=True)  # (B, K, N)
            updates = torch.einsum("bkn,bnd->bkd", attn_wm, v.float())  # (B, K, slot_dim)

            # GRU update in fp32 (autocast disabled), then MLP residual. Slots stay
            # fp32 across iterations; the to_q/to_k/to_v GEMMs re-cast under autocast.
            with torch.autocast(device_type=slots.device.type, enabled=False):
                slots = self.gru(
                    updates.reshape(-1, self.slot_dim).float(),
                    slots_prev.reshape(-1, self.slot_dim).float(),
                ).reshape(b, self.num_slots, self.slot_dim)
                slots = slots + self.mlp(self.norm_pre_ff(slots).float())

        if return_queries:
            return slots, attn, q
        return slots, attn
