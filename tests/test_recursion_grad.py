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


def test_detach_steps_keeps_forward_and_cuts_gradient():
    """detach_steps (TRM-style 1-step gradients): identical forward, but a final-step
    loss no longer reaches z0, while x (re-fed every step) still gets gradient."""
    sa, reasoner, decoder, proj, target = _setup()
    slots0 = sa(proj)[0].detach().requires_grad_(True)
    full = reasoner(slots0)["y"]
    reasoner.detach_steps = True
    cut = reasoner(slots0)["y"]
    assert all(torch.equal(a, b) for a, b in zip(full, cut))
    torch.nn.functional.mse_loss(decoder(cut[-1])[0], target).backward()
    assert reasoner.z0.grad is None
    assert slots0.grad is not None and slots0.grad.norm().item() > 0


def test_untie_steps_same_start_separate_blocks():
    """untie_steps (depth without weight sharing): steps 2..T get their own blocks, copies
    of the shared one that draw no random numbers, so a run starts from the same weights
    and data order as the tied model of the same seed. A final-step loss trains every copy."""
    torch.manual_seed(0)
    tied = TRMReasoner(slot_dim=SLOT_DIM, n_steps=T, gnn_layers=2)
    r_tied = torch.rand(1)
    torch.manual_seed(0)
    untied = TRMReasoner(slot_dim=SLOT_DIM, n_steps=T, gnn_layers=2, untie_steps=True)
    assert torch.equal(torch.rand(1), r_tied)
    assert not any(k.startswith("step_gnns") for k in tied.state_dict())  # checkpoint compat
    s = untied.state_dict()
    for k, v in tied.state_dict().items():
        assert torch.equal(s[k], v)
        if k.startswith("tiny_gnn."):
            assert all(torch.equal(s[f"step_gnns.{i}.{k[9:]}"], v) for i in range(T - 1))

    sa, reasoner, decoder, proj, target = _setup()  # nudged heads: a non-trivial recursion
    untied.load_state_dict(reasoner.state_dict(), strict=False)
    for g in untied.step_gnns:
        g.load_state_dict(reasoner.tiny_gnn.state_dict())
    slots0 = sa(proj)[0].detach()
    ys = untied(slots0)["y"]
    assert all(torch.equal(a, b) for a, b in zip(reasoner(slots0)["y"], ys))
    torch.nn.functional.mse_loss(decoder(ys[-1])[0], target).backward()
    for g in [untied.tiny_gnn, *untied.step_gnns]:
        assert g.to_y.weight.grad is not None and g.to_y.weight.grad.norm().item() > 0


def test_gnn_mlp_ratio_widens_only_the_ffn():
    """gnn_mlp_ratio (the parameter-matched shared control): the default keeps the
    original shapes (checkpoint compat); a wider ratio changes only the FFN shapes, and
    69 gives the shared block as many parameters as the untied blocks at slot_dim 256."""
    tied = TRMReasoner(slot_dim=SLOT_DIM, n_steps=T, gnn_layers=2, gnn_cross_attn=True)
    default = TRMReasoner(slot_dim=SLOT_DIM, n_steps=T, gnn_layers=2, gnn_cross_attn=True, gnn_mlp_ratio=4)
    wide = TRMReasoner(slot_dim=SLOT_DIM, n_steps=T, gnn_layers=2, gnn_cross_attn=True, gnn_mlp_ratio=69)
    shapes = lambda m: {k: v.shape for k, v in m.state_dict().items()}
    assert shapes(default) == shapes(tied)
    changed = {k for k, s in shapes(wide).items() if s != shapes(tied)[k]}
    assert changed and all(".mlp." in k for k in changed)
    assert shapes(wide).keys() == shapes(tied).keys()

    n = lambda m: sum(p.numel() for p in m.parameters())
    kw = dict(slot_dim=256, n_steps=T, gnn_layers=2, gnn_cross_attn=True)
    wide256, untied256 = TRMReasoner(**kw, gnn_mlp_ratio=69), TRMReasoner(**kw, untie_steps=True)
    assert abs(n(wide256) - n(untied256)) < 0.001 * n(untied256)


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


def test_cross_attn_grad_to_patches_and_params():
    """Step 1: with gnn_cross_attn, a synthetic `patches` tensor flows through the
    recursion — gradient must reach the cross-attn params, the patch features, and
    still reach z0 (full BPTT preserved)."""
    torch.manual_seed(0)
    reasoner = TRMReasoner(slot_dim=SLOT_DIM, n_steps=T, gnn_layers=2, gnn_cross_attn=True)
    decoder = SpatialBroadcastDecoder(slot_dim=SLOT_DIM, feature_dim=FEAT, num_patches=N)
    for head in (reasoner.tiny_gnn.to_y, reasoner.tiny_gnn.to_z):
        torch.nn.init.normal_(head.weight, std=0.1)
        torch.nn.init.normal_(head.bias, std=0.1)

    slots0 = torch.randn(2, K, SLOT_DIM)
    patches = torch.randn(2, N, SLOT_DIM, requires_grad=True)
    target = torch.randn(2, N, FEAT)

    out = reasoner(slots0, patches=patches)
    # Final-step-only loss again, so a non-zero z0 grad proves BPTT through all T steps
    # even with the cross-attn path active.
    loss = torch.nn.functional.mse_loss(decoder(out["y"][-1])[0], target)
    loss.backward()

    # The cross-attn sublayers in every block must receive gradient.
    cross_params = [
        (name, p)
        for name, p in reasoner.tiny_gnn.named_parameters()
        if "cross_attn" in name or "norm_cq" in name or "norm_ckv" in name
    ]
    assert cross_params, "cross-attn params missing -> flag not wired"
    missing = [n for n, p in cross_params if p.grad is None or p.grad.norm().item() == 0]
    assert not missing, f"cross-attn params with no gradient: {missing}"

    # Gradient flows back into the patch features (slots actually read them).
    assert patches.grad is not None and patches.grad.norm().item() > 0
    # Full BPTT still intact.
    assert reasoner.z0.grad is not None and reasoner.z0.grad.norm().item() > 0


def test_cross_attn_off_ignores_patches():
    """Default (cross_attn off) must run identically whether or not patches are passed."""
    sa, reasoner, decoder, proj, target = _setup()
    slots0, _ = sa(proj)
    patches = torch.randn(2, N, SLOT_DIM)
    out_with = reasoner(slots0, patches=patches)
    out_without = reasoner(slots0)
    assert torch.allclose(out_with["y"][-1], out_without["y"][-1])
