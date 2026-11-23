from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import re

from .contract import G1_29_JOINTS, HistoryState, RobotContract, RobotState, build_history_frame


DEPLOY_JOINTS = tuple(name + "_joint" for name in G1_29_JOINTS)


def load_bfm_config(path: str | Path) -> dict[str, Any]:
    import yaml
    with Path(path).open() as file:
        return yaml.safe_load(file)


def bfm_contract(path: str | Path) -> RobotContract:
    cfg = load_bfm_config(path)
    scale = cfg.get("action_scale", 1.0)
    scales = np.ones(29, dtype=np.float32) * scale if isinstance(scale, (int, float)) else np.array(
        [scale.get(name, 1.0) for name in DEPLOY_JOINTS], dtype=np.float32)
    defaults = cfg.get("default_joint_pos", {})
    default = np.zeros(29, dtype=np.float32)
    for i, name in enumerate(DEPLOY_JOINTS):
        if "hip_pitch" in name: default[i] = defaults.get(".*_hip_pitch_joint", -.1)
        elif "knee" in name: default[i] = defaults.get(".*_knee_joint", .3)
        elif "ankle_pitch" in name: default[i] = defaults.get(".*_ankle_pitch_joint", -.2)
    return RobotContract(joint_names=DEPLOY_JOINTS, action_scale=scales,
                         history_length=4, actor_dim=721, checkpoint=None)


def deploy_joint_values(path: str | Path) -> dict[str, np.ndarray]:
    cfg = load_bfm_config(path)
    def expand(mapping, default):
        out=[]
        for name in DEPLOY_JOINTS:
            value=default
            for pattern, val in (mapping or {}).items():
                if re.fullmatch(pattern, name): value=float(val)
            out.append(value)
        return np.asarray(out, dtype=np.float64)
    values = {"default": expand(cfg.get("default_joint_pos"), 0.),
              "kp": expand(cfg.get("joint_kp"), 40.),
              "kd": expand(cfg.get("joint_kd"), 1.),
              "action_scale": expand(cfg.get("action_scale"), 1.),
              "action_rescale": np.asarray(float(cfg.get("action_rescale", 1.)), dtype=np.float64)}
    # The deploy file contains policy gains/scales; physical limits are kept in
    # the official G1 config next to it and are mapped by exact joint name.
    limits_path = Path(path).with_name("g1.yaml")
    if limits_path.exists():
        limits = load_bfm_config(limits_path)
        for key, fallback in (("joint_pos_lower_limit", -np.inf),
                              ("joint_pos_upper_limit", np.inf),
                              ("joint_velocity_limit", np.inf),
                              ("joint_effort_limit", np.inf)):
            values[key] = expand(limits.get(key), fallback)
    else:
        values.update({"joint_pos_lower_limit": np.full(29, -np.inf),
                       "joint_pos_upper_limit": np.full(29, np.inf),
                       "joint_velocity_limit": np.full(29, np.inf),
                       "joint_effort_limit": np.full(29, np.inf)})
    return values


class BFMOnnxPolicy:
    """Minimal official deploy ONNX adapter; latent/context is explicit."""

    def __init__(self, model_path: str | Path, input_dim: int = 721):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        shape = self.session.get_inputs()[0].shape
        if shape[-1] != input_dim:
            raise ValueError(f"checkpoint input is {shape[-1]}, expected official adapter dim {input_dim}")

    def __call__(self, actor_obs: np.ndarray) -> np.ndarray:
        actor_obs = np.asarray(actor_obs, dtype=np.float32).reshape(1, -1)
        action = self.session.run([self.output_name], {self.input_name: actor_obs})[0]
        return np.asarray(action[0], dtype=np.float32)


def load_reward_latent(path: str | Path, name: str = "move-ego-0-0") -> np.ndarray:
    import pickle
    with Path(path).open("rb") as file:
        values = pickle.load(file)
    if name not in values:
        raise KeyError(f"latent {name!r} not found; available examples: {list(values)[:5]}")
    array = np.asarray(values[name], dtype=np.float32)
    return array.reshape(-1, 256)[0]


def load_policy_context(root: str | Path, task: str, name: str | None = None):
    """Load official tracking/goal context; names are never silently substituted."""
    root = Path(root) / "model"
    try:
        import joblib
        load = joblib.load
    except ImportError:
        # These published files are pickle-compatible. Keeping a stdlib
        # fallback makes shape validation usable in the minimal MuJoCo env.
        import pickle
        load = lambda path: pickle.load(path.open("rb"))
    if task == "tracking":
        values = load(root / "tracking_inference" / "zs_walking.pkl")
        return np.asarray(values, dtype=np.float32).reshape(-1, 256), "zs_walking"
    if task == "goal":
        values = load(root / "goal_inference" / "goal_reaching.pkl")
        if not name or name not in values:
            raise KeyError(f"goal {name!r} unavailable; choose one of {sorted(values)}")
        return np.asarray(values[name], dtype=np.float32).reshape(-1, 256)[0], name
    if task == "reward":
        return load_reward_latent(root / "reward_inference" / "reward_locomotion.pkl", name or "move-ego-0-0"), name or "move-ego-0-0"
    raise ValueError("policy task must be tracking, goal or reward")


def build_actor_input(state: RobotState, action: np.ndarray, history: HistoryState,
                      *, latent: np.ndarray | None = None) -> np.ndarray:
    """Build the same official 721-vector used by both active backends."""
    if latent is None:
        raise ValueError("official BFM actor input requires an explicit 256-dimensional latent")
    return build_official_actor_input(
        state.q, state.dq, state.base_quat_wxyz, state.base_angvel,
        action, history, latent)


def build_official_actor_input(q, dq, quat_wxyz, angvel_b, previous_action,
                               history, z, *, sensor_noise_std=0.0, rng=None):
    """Construct the official deploy observation used by motivo_newG1.yaml.

    Field order follows the public deploy observation config: current
    dof-pos-minus-default, dof-velocity, projected gravity, base angular
    velocity, previous action, then four newest-first histories for previous
    action, angular velocity, dof position and dof velocity, projected gravity,
    followed by a 256-dimensional context latent.
    """
    q=np.asarray(q,dtype=np.float32).reshape(-1); dq=np.asarray(dq,dtype=np.float32).reshape(-1)
    quat=np.asarray(quat_wxyz,dtype=np.float32).reshape(-1); w=np.asarray(angvel_b,dtype=np.float32).reshape(-1)
    action=np.asarray(previous_action,dtype=np.float32).reshape(-1); z=np.asarray(z,dtype=np.float32).reshape(-1)
    if q.shape != (29,) or dq.shape != (29,) or action.shape != (29,) or quat.shape != (4,) or w.shape != (3,):
        raise ValueError(f"invalid current state shapes q={q.shape} dq={dq.shape} quat={quat.shape} angvel={w.shape} action={action.shape}")
    if z.shape != (256,):
        raise ValueError(f"latent must be one (256,) vector, got {z.shape}")
    if not all(np.isfinite(x).all() for x in (q, dq, quat, w, action, z)):
        raise ValueError("non-finite BFM observation input")
    g=np.array([0,0,-1],dtype=np.float32)
    # wxyz quaternion inverse rotation, matching deploy's projected_gravity.
    qw,qx,qy,qz=quat; qv=np.array([qx,qy,qz],dtype=np.float32)
    norm = np.linalg.norm(quat)
    if not np.isclose(norm, 1.0, atol=1e-5):
        raise ValueError("base quaternion must be unit length")
    gravity=g + 2*np.cross(qv, np.cross(qv,g)-qw*g)
    pos=q-DEFAULT_POS
    w = .25*w
    # Noise belongs to measured state, never latent or remembered actions.
    # Store the same noisy measurements in history as in the current frame.
    if sensor_noise_std:
        if sensor_noise_std < 0:
            raise ValueError("sensor noise standard deviation must be nonnegative")
        rng = np.random.default_rng() if rng is None else rng
        pos = pos + rng.normal(0, sensor_noise_std, 29).astype(np.float32)
        dq = dq + rng.normal(0, sensor_noise_std, 29).astype(np.float32)
        gravity = gravity + rng.normal(0, sensor_noise_std, 3).astype(np.float32)
        w = w + rng.normal(0, sensor_noise_std, 3).astype(np.float32)
    frame=np.concatenate((pos,dq,gravity,w,action))
    hist_action=history.push_field('action',action)
    hist_w=history.push_field('angvel',w)
    hist_pos=history.push_field('pos',pos)
    hist_vel=history.push_field('vel',dq)
    hist_g=history.push_field('gravity',gravity)
    # Current state=93; history_actor=4*(29+3+29+29+3)=372; latent=256.
    obs=np.concatenate((frame,hist_action,hist_w,hist_pos,hist_vel,hist_g,z)).astype(np.float32)
    if obs.shape != (721,):
        raise ValueError(f"official actor input has shape {obs.shape}; fields={[frame.shape,hist_action.shape,hist_w.shape,hist_pos.shape,hist_vel.shape,hist_g.shape,z.shape]}")
    return obs


DEFAULT_POS=np.array((-.1,0,0,.3,-.2,0,-.1,0,0,.3,-.2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),dtype=np.float32)


def tracking_context(contexts, index, *, gamma=0.8, window_size=3):
    """Fixed official walking.yaml context: truncated future discounted mean.

    The window does not wrap at the end of the recorded motion. Its direction
    is normalized to the first context's norm, as official deploy does.
    """
    contexts = np.asarray(contexts, dtype=np.float32)
    index = int(index)
    if contexts.ndim != 2 or contexts.shape[1] != 256 or not 0 <= index < len(contexts):
        raise ValueError("invalid tracking context sequence/index")
    if window_size < 1 or not 0 < gamma <= 1:
        raise ValueError("invalid tracking context window/discount")
    window = contexts[index:index + window_size]
    discounts = gamma ** np.arange(len(window))
    discounts /= discounts.sum()
    average = np.sum(window * discounts[:, None], axis=0)
    norm = np.linalg.norm(average)
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("tracking context has invalid averaged norm")
    return (average / norm * np.linalg.norm(contexts[0])).astype(np.float32)


def next_tracking_cursor(index, active, sequence_length, *, end=2000):
    """Official walking start/end/stop state after computing the current z.

    CLI auto-start is equivalent to pressing start once. Reset starts a new
    episode; at end the policy freezes at stop=0 until the next reset.
    """
    if active and index < end:
        return (int(index) + 1) % sequence_length, True
    return 0, False
