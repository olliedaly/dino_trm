"""DINOSAUR-style spatial-broadcast MLP decoder.

Each slot is broadcast to all N patch positions, given a learned per-position
embedding, and decoded independently by a shared MLP into (feature, alpha). Alphas
are softmax-normalised across slots to combine per-slot reconstructions into the
final feature map. This is the feature-reconstruction objective from DINOSAUR
(Seitzer et al. 2023) — we reconstruct frozen backbone features, not pixels.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialBroadcastDecoder(nn.Module):
    def __init__(
        self,
        slot_dim: int = 256,
        feature_dim: int = 384,
        num_patches: int = 256,
        hidden_dim: int = 1024,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.feature_dim = feature_dim

        # Learned positional embedding, one vector per patch position.
        self.pos_emb = nn.Parameter(torch.randn(1, 1, num_patches, slot_dim) * 0.02)

        layers: list[nn.Module] = []
        in_dim = slot_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        # Last layer outputs feature_dim + 1 (the +1 is the alpha mask logit).
        layers += [nn.Linear(in_dim, feature_dim + 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (B, K, slot_dim) -> (recon (B, N, feat_dim), masks (B, K, N))."""
        b, k, _ = slots.shape
        # Broadcast each slot across all patch positions, add position embedding.
        x = slots.unsqueeze(2).expand(b, k, self.num_patches, slots.shape[-1])
        x = x + self.pos_emb
        out = self.mlp(x)  # (B, K, N, feat_dim + 1)

        recon_k, alpha_logits = out.split([self.feature_dim, 1], dim=-1)
        masks = alpha_logits.softmax(dim=1)  # softmax over slots -> (B, K, N, 1)
        recon = (recon_k * masks).sum(dim=1)  # (B, N, feat_dim)
        masks = masks.squeeze(-1)  # (B, K, N)
        return recon, masks
