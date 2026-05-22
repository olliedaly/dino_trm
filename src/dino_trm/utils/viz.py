"""Visualisation helpers for slot masks (debugging + the headline qualitative figure).

Two figures:
  * ``slot_masks_figure``: original image + each slot's mask, for one recursion step.
  * ``recursion_evolution_figure``: a grid of slot masks across recursion steps t=0..T,
    the qualitative story showing binding sharpen with reasoning depth.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data.pascal_voc import IMAGENET_MEAN, IMAGENET_STD


def unnormalize(pixel_values: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalised tensor -> (H, W, 3) uint8 image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (pixel_values.detach().cpu().float() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _masks_to_grid(masks: torch.Tensor, grid_size: int) -> np.ndarray:
    """(K, N) -> (K, grid, grid) numpy."""
    k = masks.shape[0]
    return masks.detach().cpu().float().view(k, grid_size, grid_size).numpy()


def slot_masks_figure(pixel_values: torch.Tensor, masks: torch.Tensor, grid_size: int):
    """One image + its K slot masks."""
    img = unnormalize(pixel_values)
    grid = _masks_to_grid(masks, grid_size)
    k = grid.shape[0]
    fig, axes = plt.subplots(1, k + 1, figsize=(2 * (k + 1), 2))
    axes[0].imshow(img)
    axes[0].set_title("image")
    axes[0].axis("off")
    for i in range(k):
        axes[i + 1].imshow(grid[i], cmap="viridis", vmin=0, vmax=1)
        axes[i + 1].set_title(f"slot {i}")
        axes[i + 1].axis("off")
    fig.tight_layout()
    return fig


def recursion_evolution_figure(
    pixel_values: torch.Tensor, masks_list: list[torch.Tensor], grid_size: int
):
    """Rows = recursion steps, columns = slots. masks_list: per-step (K, N)."""
    n_steps = len(masks_list)
    k = masks_list[0].shape[0]
    fig, axes = plt.subplots(n_steps, k + 1, figsize=(2 * (k + 1), 2 * n_steps))
    if n_steps == 1:
        axes = axes[None, :]
    img = unnormalize(pixel_values)
    for t in range(n_steps):
        axes[t, 0].imshow(img)
        axes[t, 0].set_ylabel(f"t={t}", fontsize=9)
        axes[t, 0].set_xticks([])
        axes[t, 0].set_yticks([])
        grid = _masks_to_grid(masks_list[t], grid_size)
        for i in range(k):
            axes[t, i + 1].imshow(grid[i], cmap="viridis", vmin=0, vmax=1)
            axes[t, i + 1].axis("off")
            if t == 0:
                axes[t, i + 1].set_title(f"slot {i}", fontsize=9)
    fig.tight_layout()
    return fig
