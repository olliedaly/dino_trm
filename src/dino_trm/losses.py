"""Training losses.

Phase 1: DINOSAUR feature-reconstruction loss (MSE between decoded and frozen
backbone features). Phase 2 adds deep-supervision weighting across recursion steps
and an ACT halting loss. Phase 3 adds an attention-entropy regulariser to discourage
slot collapse.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def feature_reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-squared error over patches and feature dim. Both (B, N, D)."""
    return F.mse_loss(recon.float(), target.float())


def deep_supervision_weights(
    num_steps: int,
    final_weight: float = 1.0,
    intermediate_weight: float = 0.25,
    device=None,
) -> torch.Tensor:
    """Per-step loss weights: small on intermediate steps, full on the final step.

    Returns a (num_steps,) tensor; index t weights the loss computed from y_{t+1}.
    """
    w = torch.full((num_steps,), intermediate_weight, dtype=torch.float32, device=device)
    w[-1] = final_weight
    return w


def per_sample_recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE per sample (mean over patches + feature dim). Both (B, N, D) -> (B,)."""
    return (recon.float() - target.float()).pow(2).mean(dim=(1, 2))


def act_halting_loss(
    halts: list[torch.Tensor], per_step_recon: list[torch.Tensor]
) -> torch.Tensor:
    """ACT-style halting loss.

    Treat halting as a per-sample classification of *which recursion step
    reconstructs best*: the target step is the argmin of per-step reconstruction
    loss, and the halting logits across steps are trained with cross-entropy toward
    it. Logits/targets are computed in fp32. ``halts`` are (B, 1) per step.
    """
    logits = torch.cat([h.float() for h in halts], dim=1)          # (B, T)
    recon = torch.stack([r.detach().float() for r in per_step_recon], dim=1)  # (B, T)
    target = recon.argmin(dim=1)                                   # (B,)
    return F.cross_entropy(logits, target)


def attention_entropy_reg(masks: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Encourage each patch to be explained by few slots (low entropy over slots).

    ``masks`` (B, K, N) are per-slot attention weights; we treat the distribution
    over the K slots at each patch and penalise its mean entropy. Minimising this
    discourages the degenerate all-slots-everywhere collapse.
    """
    p = masks.transpose(1, 2)  # (B, N, K)
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    entropy = -(p * (p + eps).log()).sum(dim=-1)  # (B, N)
    return entropy.mean()


def query_orthogonality_loss(
    queries, margin: float = 0.0, eps: float = 1e-8
) -> torch.Tensor:
    """Force slot queries to keep a minimum cosine distance from each other.

    This treats the *root cause* of slot collapse in the coupled loop: z-conditioning
    tends to drive the K slot queries toward the same vector, so every slot binds the
    same region. We penalise the off-diagonal cosine similarities of the queries
    (computed before the attention softmax), pushing them toward orthogonality
    (cosine distance 1). With ``margin > 0`` only similarities above ``1 - margin``...
    i.e. distances below ``margin``... are penalised (a hinge), leaving well-separated
    queries untouched.

    ``queries`` is a (B, K, D) tensor or a list of them (one per binding step); a list
    is averaged.
    """
    if isinstance(queries, (list, tuple)):
        if not queries:
            return torch.zeros((), requires_grad=True)
        return sum(query_orthogonality_loss(q, margin, eps) for q in queries) / len(queries)

    q = F.normalize(queries.float(), dim=-1, eps=eps)        # (B, K, D), unit rows
    sim = torch.einsum("bkd,bjd->bkj", q, q)                  # (B, K, K) cosine sims
    k = q.shape[1]
    off = ~torch.eye(k, dtype=torch.bool, device=q.device)   # off-diagonal mask
    sim_off = sim[:, off]                                     # (B, K*(K-1))
    if margin > 0:
        # Penalise only pairs closer than the margin (cosine distance < margin).
        sim_off = torch.relu(sim_off - (1.0 - margin))
    return sim_off.pow(2).mean()


def slot_balance_loss(masks: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Anti-collapse load-balancing loss on the *marginal* slot usage.

    Slot collapse = one slot explains every patch. To prevent it we keep the average
    mask mass per slot (marginalised over batch + patches) uniform: returns
    ``log(K) - H(usage)``, which is >= 0 and zero exactly when all K slots are used
    equally. NOTE this is the opposite of ``attention_entropy_reg`` (which sharpens
    per-patch assignments and, if over-weighted, *encourages* collapse).
    """
    k = masks.shape[1]
    usage = masks.mean(dim=(0, 2))               # (K,) mean mass per slot
    usage = usage / (usage.sum() + eps)
    entropy = -(usage * (usage + eps).log()).sum()
    return math.log(k) - entropy


def total_training_loss(
    out: dict,
    intermediate_weight: float = 0.25,
    final_weight: float = 1.0,
    entropy_weight: float = 0.0,
    act_weight: float = 0.0,
    balance_weight: float = 0.0,
    query_ortho_weight: float = 0.0,
    query_ortho_margin: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Assemble the full objective from a model output dict.

    Deep supervision: weighted sum of per-step feature-reconstruction losses (small
    weight on intermediate steps, full on the final step). Optionally adds an
    attention-entropy regulariser (on the final-step masks) and the ACT halting loss.
    Returns (loss, logs) where logs holds scalars for wandb.
    """
    recons = out["recons"]
    target = out["target"]
    n_steps = len(recons)
    weights = deep_supervision_weights(n_steps, final_weight, intermediate_weight)

    step_losses = [feature_reconstruction_loss(r, target) for r in recons]
    recon_loss = sum(w * l for w, l in zip(weights, step_losses))

    logs: dict[str, float] = {
        "loss/recon": float(recon_loss.detach()),
        "loss/recon_final": float(step_losses[-1].detach()),
    }
    for t, l in enumerate(step_losses):
        logs[f"loss/recon_step_{t}"] = float(l.detach())

    total = recon_loss

    if entropy_weight > 0:
        ent = attention_entropy_reg(out["masks_list"][-1])
        total = total + entropy_weight * ent
        logs["loss/entropy"] = float(ent.detach())

    if act_weight > 0 and out.get("halts"):
        per_step = [per_sample_recon_loss(r, target) for r in recons]
        act = act_halting_loss(out["halts"], per_step)
        total = total + act_weight * act
        logs["loss/act"] = float(act.detach())

    if balance_weight > 0:
        bal = slot_balance_loss(out["masks_list"][-1])
        total = total + balance_weight * bal
        logs["loss/balance"] = float(bal.detach())

    if query_ortho_weight > 0 and out.get("queries"):
        qo = query_orthogonality_loss(out["queries"], margin=query_ortho_margin)
        total = total + query_ortho_weight * qo
        logs["loss/query_ortho"] = float(qo.detach())

    logs["loss/total"] = float(total.detach())
    return total, logs
