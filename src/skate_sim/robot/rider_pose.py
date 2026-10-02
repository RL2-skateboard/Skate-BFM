"""Flat-foot rider initialization solved against the compiled G1 kinematics.

No collision margins, geometry or solver settings are changed. The 0.5 mm
sphere clearance also clears the visual sole, which extends 0.409 mm below
those spheres on the vendored G1.
"""
import mujoco
import numpy as np


def initialize_rider_pose(model, data, preset):
    mujoco.mj_forward(model, data)
    top = float(data.site('deck_top').xpos[2])
    push = preset == 'push_start'
    yaw = 0.0 if push else np.pi / 2
    data.qpos[:3] = (0, .12, .75) if push else (0, -.035, top + .70)
    data.qpos[3:7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    targets = ([[-.02, .25, .0348], [.02, 0, top + .0348]] if push else
               [[-.16, -.035, top + .0348], [.16, -.035, top + .0348]])
    for side, target in zip(('left', 'right'), targets):
        names = [f'{side}_{part}_joint' for part in
                 ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
        ids = [model.joint(name).id for name in names]
        addresses = model.jnt_qposadr[ids]
        limits = model.jnt_range[ids]
        body = model.body(f'{side}_ankle_roll_link').id

        def residual(q):
            data.qpos[addresses] = q
            mujoco.mj_forward(model, data)
            return np.r_[data.xpos[body] - target,
                         .2 * (data.xmat[body].reshape(3, 3) - rotation).ravel()]

        q = np.array([-.3, 0, 0, .6, -.3, 0])
        for _ in range(100):
            error = residual(q)
            if np.linalg.norm(error) < 1e-8:
                break
            jac = np.column_stack([(residual(q + np.eye(6)[i] * 1e-6) - error) / 1e-6
                                   for i in range(6)])
            delta = np.linalg.solve(jac.T @ jac + 1e-6 * np.eye(6), -jac.T @ error)
            q = np.clip(q + np.clip(delta, -.15, .15), limits[:, 0], limits[:, 1])
        if np.linalg.norm(residual(q)) > 1e-6:
            raise ValueError(f'Cannot initialize {preset}: {side} flat-foot IK did not converge')
    mujoco.mj_forward(model, data)
