"""MLPBlockReasoner: param-matched non-recursive control for TRM.

Mirrors the slim grad-flow tests for TRMReasoner — same synthetic-patches pattern,
no backbone — so the tests run in seconds on CPU.
"""

import torch

from dino_trm.models.decoder import SpatialBroadcastDecoder
from dino_trm.models.mlp_block import MLPBlockReasoner
from dino_trm.models.trm_module import TRMReasoner

SLOT_DIM = 64
K = 7
N = 20
FEAT = 48


def _decoder():
    return SpatialBroadcastDecoder(slot_dim=SLOT_DIM, feature_dim=FEAT, num_patches=N)


def test_shape_and_residual_identity_at_init():
    """to_y is zero-initialised, so the un-trained reasoner must be the identity."""
    torch.manual_seed(0)
    reasoner = MLPBlockReasoner(slot_dim=SLOT_DIM, n_layers=3)
    slots = torch.randn(2, K, SLOT_DIM)
    out = reasoner(slots)
    assert out.shape == slots.shape
    assert torch.allclose(out, slots), "block must start as identity (zero-init to_y)"


def test_all_params_get_grad():
    """Every block + readout parameter receives a non-zero gradient."""
    torch.manual_seed(0)
    reasoner = MLPBlockReasoner(slot_dim=SLOT_DIM, n_layers=3)
    # Nudge to_y off zero so the block is non-degenerate (mirrors test_recursion_grad).
    torch.nn.init.normal_(reasoner.to_y.weight, std=0.1)
    torch.nn.init.normal_(reasoner.to_y.bias, std=0.1)
    decoder = _decoder()

    slots = torch.randn(2, K, SLOT_DIM)
    target = torch.randn(2, N, FEAT)
    out = reasoner(slots)
    loss = torch.nn.functional.mse_loss(decoder(out)[0], target)
    loss.backward()

    missing = [
        name
        for name, p in reasoner.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.norm().item() == 0)
    ]
    assert not missing, f"params with no gradient: {missing}"


def test_cross_attn_consumes_patches():
    """With cross_attn=True, gradient must reach the patch features."""
    torch.manual_seed(0)
    reasoner = MLPBlockReasoner(slot_dim=SLOT_DIM, n_layers=2, cross_attn=True)
    torch.nn.init.normal_(reasoner.to_y.weight, std=0.1)
    decoder = _decoder()

    slots = torch.randn(2, K, SLOT_DIM)
    patches = torch.randn(2, N, SLOT_DIM, requires_grad=True)
    target = torch.randn(2, N, FEAT)
    out = reasoner(slots, patches=patches)
    loss = torch.nn.functional.mse_loss(decoder(out)[0], target)
    loss.backward()

    assert patches.grad is not None and patches.grad.norm().item() > 0
    cross_params = [
        (n, p) for n, p in reasoner.named_parameters() if "cross_attn" in n or "norm_c" in n
    ]
    assert cross_params, "cross-attn params missing"
    missing = [n for n, p in cross_params if p.grad is None or p.grad.norm().item() == 0]
    assert not missing, f"cross-attn params with no gradient: {missing}"


def test_param_count_meets_or_exceeds_trm_reasoner():
    """The control must have at least as many trainable params as the recursive
    reasoner it replaces — otherwise a 'fewer params' confound creeps in. Compared
    at matched cross-attn setting and slot/head/layer geometry."""
    trm = TRMReasoner(slot_dim=SLOT_DIM, n_steps=8, gnn_layers=2, gnn_heads=4, gnn_cross_attn=True)
    mlp = MLPBlockReasoner(slot_dim=SLOT_DIM, n_layers=3, n_heads=4, cross_attn=True)
    trm_n = sum(p.numel() for p in trm.parameters() if p.requires_grad)
    mlp_n = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    assert mlp_n >= trm_n, f"control has fewer params than TRM: {mlp_n} < {trm_n}"
