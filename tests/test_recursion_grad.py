"""CRITICAL: verify gradients flow through the entire T-step recursion (full BPTT).

The most common bug in this kind of model is silently truncating the gradient (e.g.
HRM's 1-step approximation), which TRM showed badly hurts. These tests use synthetic
slot features (no backbone) so they run fast on CPU.

Key signal: ``z0`` (the learned latent init) is used ONLY at step 0. If a loss that
depends solely on the FINAL step still produces a non-zero gradient on ``z0``, the
gradient must have traversed all T steps — i.e. BPTT is intact end-to-end.
"""

import torch

from dino_trm.models.decoder import SpatialBroadcastDecoder
from dino_trm.models.slot_attention import SlotAttention
from dino_trm.models.trm_module import TRMReasoner

SLOT_DIM = 64
K = 7
N = 20
FEAT = 48
T = 8


def _setup():
    """Build the synthetic pieces, nudging the zero-initialised residual heads off
    zero so the recursion is non-degenerate — i.e. representative of the trained
    network, where full-BPTT gradient flow is what we care about. (At init the
    ReZero-style heads make the recursion an identity map, which trivially carries
    no gradient to the latent init.)"""
    torch.manual_seed(0)
    sa = SlotAttention(num_slots=K, slot_dim=SLOT_DIM, input_dim=SLOT_DIM, n_iters=3)
    reasoner = TRMReasoner(slot_dim=SLOT_DIM, n_steps=T, gnn_layers=2)
    decoder = SpatialBroadcastDecoder(slot_dim=SLOT_DIM, feature_dim=FEAT, num_patches=N)
    for head in (reasoner.tiny_gnn.to_y, reasoner.tiny_gnn.to_z):
        torch.nn.init.normal_(head.weight, std=0.1)
        torch.nn.init.normal_(head.bias, std=0.1)
    proj = torch.randn(2, N, SLOT_DIM)
    target = torch.randn(2, N, FEAT)
    return sa, reasoner, decoder, proj, target


def test_z0_grad_from_final_step_only():
    """Loss on final step alone must still reach z0 -> full BPTT across all T steps."""
    sa, reasoner, decoder, proj, target = _setup()
    slots0, _ = sa(proj)
    out = reasoner(slots0)
    recon_final, _ = decoder(out["y"][-1])
    loss = torch.nn.functional.mse_loss(recon_final, target)
    loss.backward()

    assert reasoner.z0.grad is not None
    assert reasoner.z0.grad.norm().item() > 0, "z0 got no gradient -> recursion truncated"


def test_all_recursion_params_get_grad():
    """Every TinyGNN + halt-head parameter receives a non-zero gradient."""
    sa, reasoner, decoder, proj, target = _setup()
    slots0, _ = sa(proj)
    out = reasoner(slots0)
    # Deep supervision over all steps, plus a halt term so the halting head (which
    # the reconstruction loss alone doesn't touch) also receives gradient.
    loss = sum(
        torch.nn.functional.mse_loss(decoder(y)[0], target) for y in out["y"]
    )
    loss = loss + sum(h.float().pow(2).mean() for h in out["halt"])
    loss.backward()

    missing = [
        name
        for name, p in reasoner.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.norm().item() == 0)
    ]
    assert not missing, f"params with no gradient: {missing}"


def test_full_bptt_differs_from_truncated():
    """Detaching z between steps must change z0's gradient, proving BPTT is real."""
    sa, reasoner, decoder, proj, target = _setup()

    # Detach slots0 so the two branches (full vs truncated) don't share a graph and
    # the comparison isolates gradient flow through the recursion's latent path.
    slots0, _ = sa(proj)
    slots0 = slots0.detach()
    out = reasoner(slots0)
    loss = torch.nn.functional.mse_loss(decoder(out["y"][-1])[0], target)
    loss.backward()
    full_grad = reasoner.z0.grad.clone()

    # Truncated reference: detach the latent each step (HRM-style 1-step approx).
    reasoner.zero_grad()
    y = slots0
    z = reasoner.init_latent(slots0)
    for _ in range(T):
        y, z = reasoner.tiny_gnn(slots0, y, z)
        z = z.detach()
    loss_trunc = torch.nn.functional.mse_loss(decoder(y)[0], target)
    loss_trunc.backward()
    trunc_grad = reasoner.z0.grad.clone()

    # Truncating the latent path should leave z0 with (near) zero gradient.
    assert trunc_grad.norm().item() < full_grad.norm().item()


def test_coupled_rebind_grad_to_slot_attention():
    """In coupled mode, gradient reaches slot-attention params via the rebind path."""
    sa, reasoner, decoder, proj, target = _setup()
    # Make z-conditioning non-trivial so the feedback path carries gradient.
    torch.nn.init.normal_(sa.z_to_slot.weight, std=0.3)

    slots0, _ = sa(proj)

    def rebind(z):
        slots, _ = sa(proj, z_prev=z)
        return slots

    out = reasoner(slots0, rebind=rebind)
    loss = torch.nn.functional.mse_loss(decoder(out["y"][-1])[0], target)
    loss.backward()

    assert sa.z_to_slot.weight.grad is not None
    assert sa.z_to_slot.weight.grad.norm().item() > 0
