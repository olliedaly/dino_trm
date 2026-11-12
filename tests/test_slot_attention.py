import torch

from dino_trm.models.decoder import SpatialBroadcastDecoder
from dino_trm.models.slot_attention import SlotAttention


def test_slot_attention_shapes():
    torch.manual_seed(0)
    sa = SlotAttention(num_slots=7, slot_dim=64, input_dim=64, n_iters=3)
    x = torch.randn(2, 20, 64)
    slots, attn = sa(x)
    assert slots.shape == (2, 7, 64)
    assert attn.shape == (2, 7, 20)
    # Attention is a softmax over slots (slots compete per input) -> sums to 1 over K.
    assert torch.allclose(attn.sum(dim=1), torch.ones(2, 20), atol=1e-4)


def test_z_conditioning_changes_output():
    torch.manual_seed(0)
    sa = SlotAttention(num_slots=7, slot_dim=64, input_dim=64, n_iters=3)
    # Force the (zero-init) z projection to be non-trivial so the hook is exercised.
    torch.nn.init.normal_(sa.z_to_slot.weight, std=0.5)
    x = torch.randn(2, 20, 64)
    z = torch.randn(2, 7, 64)
    g = torch.Generator().manual_seed(123)
    torch.manual_seed(1)
    s_plain, _ = sa(x)
    torch.manual_seed(1)
    s_cond, _ = sa(x, z_prev=z)
    assert not torch.allclose(s_plain, s_cond)


def test_return_all_matches_final_iteration():
    torch.manual_seed(0)
    sa = SlotAttention(num_slots=7, slot_dim=64, input_dim=64, n_iters=4)
    x = torch.randn(2, 20, 64)
    torch.manual_seed(1)
    slots, attn = sa(x)
    torch.manual_seed(1)
    slots_all, attn_all, q = sa(x, return_all=True)
    assert len(slots_all) == len(attn_all) == 4
    assert torch.equal(slots_all[-1], slots)
    assert torch.equal(attn_all[-1], attn)
    assert q.shape == (2, 7, 64)


def test_final_iteration_loss_reaches_slot_init():
    """A loss on the last iteration alone must reach the slot initialisation, which is
    used only before iteration 1 -> backprop runs through every iteration (sa_rec)."""
    torch.manual_seed(0)
    sa = SlotAttention(num_slots=7, slot_dim=64, input_dim=64, n_iters=8)
    slots_all, _, _ = sa(torch.randn(2, 20, 64), return_all=True)
    slots_all[-1].pow(2).mean().backward()
    assert sa.slots_mu.grad is not None
    assert sa.slots_mu.grad.norm().item() > 0


def test_detach_iters_keeps_forward_and_cuts_gradient():
    """detach_iters: same slots, but a loss on the last iteration trains only that
    iteration's update, so it no longer reaches the slot initialisation."""
    torch.manual_seed(0)
    sa = SlotAttention(num_slots=7, slot_dim=64, input_dim=64, n_iters=8)
    x = torch.randn(2, 20, 64)
    torch.manual_seed(1)
    full, _, _ = sa(x, return_all=True)
    torch.manual_seed(1)
    cut, _, _ = sa(x, return_all=True, detach_iters=True)
    assert all(torch.equal(a, b) for a, b in zip(full, cut))
    cut[-1].pow(2).mean().backward()
    assert sa.slots_mu.grad is None
    assert sa.gru.weight_ih.grad.norm().item() > 0


def test_decoder_masks_normalised():
    torch.manual_seed(0)
    dec = SpatialBroadcastDecoder(slot_dim=64, feature_dim=48, num_patches=20)
    slots = torch.randn(2, 7, 64)
    recon, masks = dec(slots)
    assert recon.shape == (2, 20, 48)
    assert masks.shape == (2, 7, 20)
    # Alpha masks are a softmax over slots -> sum to 1 over K at each patch.
    assert torch.allclose(masks.sum(dim=1), torch.ones(2, 20), atol=1e-4)


def test_decoder_loc_bias_makes_masks_local():
    torch.manual_seed(0)
    dec = SpatialBroadcastDecoder(slot_dim=64, feature_dim=48, num_patches=16)
    torch.nn.init.normal_(dec.pos_emb, std=1.0)  # masks that vary over space
    loc = SpatialBroadcastDecoder(slot_dim=64, feature_dim=48, num_patches=16, loc_bias=20.0)
    loc.load_state_dict(dec.state_dict())  # the prior adds no saved weights
    slots = torch.randn(2, 7, 64)
    _, m0 = dec(slots)
    _, m1 = loc(slots)
    assert torch.allclose(m1.sum(dim=1), torch.ones(2, 16), atol=1e-4)

    def spread(m):  # mask-weighted squared distance from each slot's centroid
        w = m.unsqueeze(-1)
        c = (w * loc.patch_pos).sum(2, keepdim=True) / w.sum(2, keepdim=True)
        return ((loc.patch_pos - c).pow(2).sum(-1) * m).sum() / m.sum()

    assert spread(m1) < spread(m0)


def test_sample_passes_gives_each_image_one_pass_in_proportion_to_its_weight():
    """Intermediate passes have weight 0.25 and the final pass 1.0, so the final pass is
    drawn 1 / 2.75 = 36% of the time and each of the other seven 9%."""
    from dino_trm.models.full_model import sample_passes
    torch.manual_seed(0)
    assert sample_passes(8, 32).shape == (32,)
    draws = sample_passes(8, 40000)
    assert draws.min() >= 0 and draws.max() <= 7
    freq = torch.bincount(draws, minlength=8).float() / len(draws)
    assert abs(freq[7].item() - 1 / 2.75) < 0.01
    assert all(abs(f.item() - 0.25 / 2.75) < 0.006 for f in freq[:7])


def test_sampled_pass_selection_picks_each_images_own_pass():
    """The indexing used in the model: image b gets the slots of its own sampled pass."""
    slots_all = [torch.full((4, 7, 3), float(t)) for t in range(8)]
    picked = torch.tensor([7, 0, 3, 7])
    chosen = torch.stack(slots_all)[picked, torch.arange(4)]
    assert chosen.shape == (4, 7, 3)
    assert chosen[:, 0, 0].tolist() == [7.0, 0.0, 3.0, 7.0]
