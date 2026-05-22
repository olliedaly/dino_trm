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


def test_decoder_masks_normalised():
    torch.manual_seed(0)
    dec = SpatialBroadcastDecoder(slot_dim=64, feature_dim=48, num_patches=20)
    slots = torch.randn(2, 7, 64)
    recon, masks = dec(slots)
    assert recon.shape == (2, 20, 48)
    assert masks.shape == (2, 7, 20)
    # Alpha masks are a softmax over slots -> sum to 1 over K at each patch.
    assert torch.allclose(masks.sum(dim=1), torch.ones(2, 20), atol=1e-4)
