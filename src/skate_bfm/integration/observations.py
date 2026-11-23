from __future__ import annotations

import math

import numpy as np
import torch




def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion).copy()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(np.asarray(left), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion)
    vector = np.asarray(vector)
    q_vector = quaternion[..., 1:]
    return (
        vector
        + 2.0 * quaternion[..., :1] * np.cross(q_vector, vector)
        + 2.0 * np.cross(q_vector, np.cross(q_vector, vector))
    )


def _extend_head_state(
    body_position: np.ndarray,
    body_quaternion: np.ndarray,
    body_linear_velocity: np.ndarray,
    body_angular_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torso_index = 15
    head_offset = _quat_rotate(
        body_quaternion[torso_index],
        np.asarray((0.0, 0.0, 0.35), dtype=np.float64),
    )
    head_position = body_position[torso_index] + head_offset
    head_angular_velocity = body_angular_velocity[torso_index]
    head_linear_velocity = (
        body_linear_velocity[torso_index]
        + np.cross(head_angular_velocity, head_offset)
    )
    return (
        np.concatenate((body_position, head_position[None]), axis=0),
        np.concatenate((body_quaternion, body_quaternion[torso_index : torso_index + 1]), axis=0),
        np.concatenate((body_linear_velocity, head_linear_velocity[None]), axis=0),
        np.concatenate((body_angular_velocity, head_angular_velocity[None]), axis=0),
    )


def bfm0_privileged_state(
    body_position: np.ndarray,
    body_quaternion: np.ndarray,
    body_linear_velocity: np.ndarray,
    body_angular_velocity: np.ndarray,
) -> np.ndarray:
    """Build the official 463D max-local-self observation from the 30 physical G1 bodies."""

    position, quaternion, linear_velocity, angular_velocity = _extend_head_state(
        np.asarray(body_position, dtype=np.float64),
        np.asarray(body_quaternion, dtype=np.float64),
        np.asarray(body_linear_velocity, dtype=np.float64),
        np.asarray(body_angular_velocity, dtype=np.float64),
    )
    root_forward = _quat_rotate(
        quaternion[:1],
        np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64),
    )[0]
    heading = math.atan2(root_forward[1], root_forward[0])
    heading_inverse = np.asarray(
        (math.cos(-heading / 2.0), 0.0, 0.0, math.sin(-heading / 2.0)),
        dtype=np.float64,
    )
    heading_quaternions = np.broadcast_to(heading_inverse, quaternion.shape)
    local_position = _quat_rotate(
        heading_quaternions,
        position - position[0],
    )[1:]
    local_quaternion = _quat_multiply(heading_quaternions, quaternion)
    tangent = _quat_rotate(
        local_quaternion,
        np.broadcast_to((1.0, 0.0, 0.0), position.shape),
    )
    normal = _quat_rotate(
        local_quaternion,
        np.broadcast_to((0.0, 0.0, 1.0), position.shape),
    )
    local_linear_velocity = _quat_rotate(heading_quaternions, linear_velocity)
    local_angular_velocity = _quat_rotate(heading_quaternions, angular_velocity)
    result = np.concatenate(
        (
            position[0, 2:3],
            local_position.reshape(-1),
            np.concatenate((tangent, normal), axis=-1).reshape(-1),
            local_linear_velocity.reshape(-1),
            local_angular_velocity.reshape(-1),
        )
    ).astype(np.float32)
    if result.shape != (463,):
        raise ValueError(
            f"Official BFM privileged state must have shape (463,), got {result.shape}."
        )
    return result
