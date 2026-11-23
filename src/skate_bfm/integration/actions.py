"""Native 29-DoF BFM joint ordering and actuator constants."""
from __future__ import annotations

import numpy as np

BFM0_ACTION_RESCALE = 5.0
BFM0_ACTION_SCALE = 0.25
BFM0_ACTION_CLIP = 5.0
BFM0_KP = np.asarray(
    (
        99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
        99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
        300.0, 300.0, 300.0,
        14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 16.77833, 16.77833,
        14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 16.77833, 16.77833,
    ),
    dtype=np.float64,
)
BFM0_KD = np.asarray(
    (
        6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
        6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
        5.0, 5.0, 5.0,
        0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 1.06814, 1.06814,
        0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 1.06814, 1.06814,
    ),
    dtype=np.float64,
)
BFM0_EFFORT_LIMITS = np.asarray(
    (
        139.0, 139.0, 88.0, 139.0, 50.0, 50.0,
        139.0, 139.0, 88.0, 139.0, 50.0, 50.0,
        88.0, 50.0, 50.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    ),
    dtype=np.float64,
)
BFM0_DEFAULT_JOINT_POSITION = np.asarray(
    (
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    dtype=np.float32,
)
BFM0_ACTION_TARGET_GAINS = (
    BFM0_ACTION_RESCALE * BFM0_ACTION_SCALE * BFM0_EFFORT_LIMITS / BFM0_KP
)

BFM0_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
