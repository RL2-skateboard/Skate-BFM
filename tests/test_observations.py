import numpy as np

from skate_bfm.integration import HuskyToBfm0Observation


def test_husky_observation_maps_to_bfm0_schema() -> None:
    adapter = HuskyToBfm0Observation()
    observation = adapter(
        {
            "joint_position": np.zeros(23, dtype=np.float32),
            "joint_velocity": np.zeros(23, dtype=np.float32),
            "last_action": np.zeros(23, dtype=np.float32),
            "projected_gravity": np.array((0.0, 0.0, -1.0), dtype=np.float32),
            "angular_velocity": np.zeros(3, dtype=np.float32),
        }
    )

    assert observation["state"].shape == (64,)
    assert observation["history"].shape == (372,)
    assert observation["last_action"].shape == (29,)

