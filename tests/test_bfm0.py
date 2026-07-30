import math

import torch

from skate_bfm.bfm0 import Bfm0Model


def _observation(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "state": torch.zeros(batch, 64),
        "history": torch.zeros(batch, 372),
        "last_action": torch.zeros(batch, 29),
    }


def test_bfm0_shapes_and_latent_projection() -> None:
    model = Bfm0Model()
    z = torch.randn(2, 256)
    action = model.act(_observation(), z)
    embedding = model.forward_embedding(_observation(), action, z)

    assert action.shape == (2, 29)
    assert embedding.shape == (2, 256)
    assert torch.allclose(
        torch.linalg.vector_norm(model.project_z(z), dim=-1),
        torch.full((2,), math.sqrt(256)),
        atol=1e-4,
    )


def test_reward_inference_returns_one_latent() -> None:
    model = Bfm0Model()
    z = model.infer_reward_latent(_observation(batch=4), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert z.shape == (256,)

