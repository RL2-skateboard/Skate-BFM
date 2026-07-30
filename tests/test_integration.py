import torch

from skate_bfm.integration.actions import BFM0_JOINTS, HUSKY_JOINTS, Bfm0ToHusky23


def test_joint_mapping_is_name_based() -> None:
    adapter = Bfm0ToHusky23(action_clip=None)
    source = torch.arange(29, dtype=torch.float32)
    target = adapter(source)
    source_by_name = dict(zip(BFM0_JOINTS, source, strict=True))

    assert target.shape == (23,)
    assert target.tolist() == [source_by_name[name].item() for name in HUSKY_JOINTS]
    assert adapter.mapping.dropped == (
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )


def test_joint_mapping_supports_batches_and_clipping() -> None:
    adapter = Bfm0ToHusky23(action_gain=2.0, action_clip=1.0)
    target = adapter(torch.full((3, 29), 0.75))
    assert target.shape == (3, 23)
    assert torch.all(target == 1.0)

