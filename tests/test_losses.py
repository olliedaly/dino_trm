import torch

from dino_trm.losses import (
    deep_supervision_weights,
    query_orthogonality_loss,
    slot_balance_loss,
    total_training_loss,
)


def test_query_orthogonality_loss_extremes():
    # Identical (collapsed) queries -> max penalty; orthonormal -> ~0.
    ident = torch.ones(2, 7, 16)
    assert query_orthogonality_loss(ident).item() > 0.9
    orth = torch.zeros(2, 7, 16)
    for k in range(7):
        orth[:, k, k] = 1.0
    assert query_orthogonality_loss(orth).item() < 1e-6
    # List form averages over binding steps.
    assert query_orthogonality_loss([ident, orth]).item() > 0.4


def test_query_ortho_margin_ignores_separated():
    # With a margin, well-separated (orthogonal) queries incur no penalty.
    orth = torch.zeros(2, 7, 16)
    for k in range(7):
        orth[:, k, k] = 1.0
    assert query_orthogonality_loss(orth, margin=0.5).item() < 1e-6


def test_slot_balance_loss_extremes():
    # All mass on one slot (collapse) -> high; uniform usage -> ~0.
    collapsed = torch.zeros(2, 7, 20)
    collapsed[:, 0, :] = 1.0
    uniform = torch.full((2, 7, 20), 1.0 / 7)
    assert slot_balance_loss(collapsed).item() > slot_balance_loss(uniform).item()
    assert slot_balance_loss(uniform).item() < 1e-4


def _fake_out(n_steps=4, b=2, n=20, d=48, k=7):
    target = torch.randn(b, n, d)
    recons = [torch.randn(b, n, d, requires_grad=True) for _ in range(n_steps)]
    masks = [torch.rand(b, k, n) for _ in range(n_steps)]
    halts = [torch.randn(b, 1) for _ in range(n_steps)]
    return {"target": target, "recons": recons, "masks_list": masks, "halts": halts}


def test_deep_supervision_weights():
    w = deep_supervision_weights(4, final_weight=1.0, intermediate_weight=0.25)
    assert torch.allclose(w, torch.tensor([0.25, 0.25, 0.25, 1.0]))


def test_total_loss_baseline_single_step():
    out = {
        "target": torch.randn(2, 20, 48),
        "recons": [torch.randn(2, 20, 48, requires_grad=True)],
        "masks_list": [torch.rand(2, 7, 20)],
        "halts": [],
    }
    loss, logs = total_training_loss(out)
    assert loss.requires_grad
    assert "loss/recon_final" in logs and "loss/total" in logs
    loss.backward()


def test_total_loss_with_entropy_and_act():
    out = _fake_out()
    loss, logs = total_training_loss(
        out, entropy_weight=0.01, act_weight=0.1
    )
    assert "loss/entropy" in logs
    assert "loss/act" in logs
    # Per-step recon losses logged for the metric-vs-depth story.
    assert "loss/recon_step_0" in logs and "loss/recon_step_3" in logs
    loss.backward()
