import torch

from dino_trm.models.tiny_gnn import TinyGNN


def test_tiny_gnn_shapes():
    gnn = TinyGNN(slot_dim=64, n_layers=2, n_heads=4)
    x = torch.randn(2, 7, 64)
    y = torch.randn(2, 7, 64)
    z = torch.randn(2, 7, 64)
    y2, z2 = gnn(x, y, z)
    assert y2.shape == (2, 7, 64)
    assert z2.shape == (2, 7, 64)


def test_residual_identity_at_init():
    # to_y / to_z are zero-initialised, so at init the block is an identity map on
    # both streams. This guards against accidental scale blow-ups from the new block.
    gnn = TinyGNN(slot_dim=64, n_layers=2, n_heads=4)
    x = torch.randn(2, 7, 64)
    y = torch.randn(2, 7, 64)
    z = torch.randn(2, 7, 64)
    y2, z2 = gnn(x, y, z)
    assert torch.allclose(y2, y, atol=1e-6)
    assert torch.allclose(z2, z, atol=1e-6)
