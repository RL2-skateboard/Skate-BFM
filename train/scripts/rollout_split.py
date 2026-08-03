#!/usr/bin/env python3
"""Split one HUSKY rollout into command-labelled motion clips.

The script preserves the source rollout, slices only existing frame-aligned
fields, and refuses expert export when failure detection is not reliable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import pickle
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

UNKNOWN_FAILURE_DETECTION = "UNKNOWN_FAILURE_DETECTION"
NATIVE_FAILURE_FIELDS = (
    "terminated",
    "reset_buf",
    "reset_terminated",
    "fall",
    "fallen",
    "termination",
    "termination_condition",
    "forbidden_contact",
    "illegal_contact",
)
RESET_FIELDS = ("reset_buf", "reset", "reset_flag", "resets")
TIME_FIELDS = ("sim_time", "simulation_time", "time", "timestamps")
COMMAND_FIELDS = ("high_level_command", "command", "commands", "command_history")
QPOS_FIELDS = ("qpos", "joint_qpos")
ROOT_POS_FIELDS = ("root_pos", "root_position", "base_pos", "base_position")
ROOT_QUAT_FIELDS = ("root_quat", "root_quaternion", "base_quat", "base_quaternion")

SEGMENT_FIELDS = (
    "segment_id",
    "episode_id",
    "motion_label",
    "key",
    "command",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "duration",
    "num_frames",
    "pose_path",
    "video_path",
    "status",
    "failure_detected",
    "failure_frame",
    "failure_reason",
    "truncated_by_failure",
    "original_end_frame",
    "valid_end_frame",
    "reset_detected",
    "discard_reason",
    "notes",
)


@dataclass
class Rollout:
    path: Path
    suffix: str
    payload: Any
    top_type: str
    keys: list[str]
    shape: list[int] | None
    dtype: str | None
    num_frames: int


@dataclass
class KeyEvent:
    frame_idx: int
    sim_time: float | None
    key: str
    event_type: str
    command: str | None


@dataclass
class Span:
    command: str
    key: str
    start: int
    end: int


@dataclass
class Segment:
    segment_id: str
    episode_id: str
    motion_label: str
    key: str
    command: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    num_frames: int
    pose_path: str = ""
    video_path: str = ""
    status: str = "valid"
    failure_detected: bool = False
    failure_frame: int | None = None
    failure_reason: str = ""
    truncated_by_failure: bool = False
    original_end_frame: int | None = None
    valid_end_frame: int | None = None
    reset_detected: bool = False
    discard_reason: str = ""
    notes: str = ""


class KeySegmentRecorder:
    """Minimal event recorder suitable for calling from a keyboard callback."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def on_key_event(
        self,
        key: str,
        frame_idx: int,
        sim_time: float,
        event_type: str,
        command: str | None = None,
    ) -> None:
        event_type = normalize_event_type(event_type)
        if event_type not in {"key_down", "key_up"}:
            raise ValueError(f"unsupported event_type: {event_type}")
        self.events.append(
            {
                "frame_idx": int(frame_idx),
                "sim_time": float(sim_time),
                "key": normalize_key(key),
                "event_type": event_type,
                "command": command or "",
            }
        )

    def finalize(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        events = sorted(self.events, key=lambda item: item["frame_idx"])
        if path.suffix.lower() == ".jsonl":
            with path.open("w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
            return
        if path.suffix.lower() != ".csv":
            raise ValueError("KeySegmentRecorder output must be .csv or .jsonl")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("frame_idx", "sim_time", "key", "event_type", "command"),
            )
            writer.writeheader()
            writer.writerows(events)


class OfficialPhaseClock:
    """The fixed HUSKY phase schedule used by the official environment."""

    phase_ratios = (0.0, 0.4, 0.5, 0.95, 1.0)

    def __init__(self, policy_frequency: int, cycle_time: float = 6.0) -> None:
        self.step_dt = 1.0 / policy_frequency
        self.cycle_time = cycle_time
        self.step_count = 0

    def reset(self) -> None:
        self.step_count = 0

    def next(self) -> tuple[str, float]:
        self.step_count += 1
        phase_value = (self.step_count * self.step_dt / self.cycle_time) % 1.0
        p0, p1, p2, p3, p4 = self.phase_ratios
        if p0 <= phase_value < p1:
            label = "push"
        elif p1 <= phase_value < p2:
            label = "push2steer"
        elif p2 <= phase_value < p3:
            label = "steer"
        elif p3 <= phase_value <= p4:
            label = "steer2push"
        else:
            label = "push"
        return label, phase_value


class LiveFallDetector:
    """Detect a persistent fall without treating foot lift-off as a fall."""

    _illegal_geom = re.compile(
        r"(left|right)_(shin|linkage_brace|shoulder_yaw|elbow_yaw|wrist|hand)_collision"
        r"|robot/pelvis_collision$"
    )

    def __init__(
        self,
        model: Any,
        orientation_limit_deg: float,
        root_height_min: float,
        confirm_frames: int,
    ) -> None:
        self.model = model
        self.orientation_limit_deg = orientation_limit_deg
        self.root_height_min = root_height_min
        self.confirm_frames = max(1, confirm_frames)
        self.bad_frames = 0
        self.foot_geoms: set[int] = set()
        self.board_geoms: set[int] = set()
        self.illegal_geoms: set[int] = set()
        import mujoco

        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not name:
                continue
            if re.search(r"robot/(left|right)_foot[0-9]+_collision$", name):
                self.foot_geoms.add(geom_id)
            if name == "skateboard/skateboard_deck_collision":
                self.board_geoms.add(geom_id)
            if self._illegal_geom.search(name):
                self.illegal_geoms.add(geom_id)

    def reset(self) -> None:
        self.bad_frames = 0

    def check(self, data: Any) -> tuple[bool, list[str], dict[str, Any]]:
        qw, qx, qy, qz = np.asarray(data.qpos[3:7], dtype=float)
        norm = np.linalg.norm((qw, qx, qy, qz))
        if norm <= 0.0:
            tilt_deg = 180.0
        else:
            qw, qx, qy, qz = np.asarray((qw, qx, qy, qz)) / norm
            gravity_z = 1.0 - 2.0 * (qx * qx + qy * qy)
            tilt_deg = math.degrees(math.acos(np.clip(gravity_z, -1.0, 1.0)))

        feet_on_board = False
        illegal_contact = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom_a, geom_b = contact.geom1, contact.geom2
            if (geom_a in self.foot_geoms and geom_b in self.board_geoms) or (
                geom_b in self.foot_geoms and geom_a in self.board_geoms
            ):
                feet_on_board = True
            if geom_a in self.illegal_geoms or geom_b in self.illegal_geoms:
                illegal_contact = True
        root_height = float(data.qpos[2])
        severe_tilt = tilt_deg > self.orientation_limit_deg
        low_contact_fall = root_height < self.root_height_min and illegal_contact
        candidate = severe_tilt or low_contact_fall
        self.bad_frames = self.bad_frames + 1 if candidate else 0
        reasons = []
        if severe_tilt:
            reasons.append(f"tilt>{self.orientation_limit_deg:.0f}deg")
        if low_contact_fall:
            reasons.append(f"height<{self.root_height_min:.2f}+illegal_contact")
        diagnostics = {
            "tilt_deg": tilt_deg,
            "root_height": root_height,
            "feet_on_board": feet_on_board,
            "illegal_contact": illegal_contact,
            "fall_candidate": candidate,
            "confirm_frames": self.bad_frames,
        }
        return self.bad_frames >= self.confirm_frames, reasons, diagnostics


def load_upstream_sim(robot_xml: Path) -> Any:
    sim_path = robot_xml.resolve().parent / "sim.py"
    if not sim_path.is_file():
        raise FileNotFoundError(f"official HUSKY sim.py not found beside {robot_xml}")
    spec = importlib.util.spec_from_file_location("skate_bfm_upstream_sim", sim_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load HUSKY sim.py from {sim_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_live(args: argparse.Namespace) -> int:
    """Run the existing HUSKY controller with the official phase clock."""
    robot_xml = args.robot_xml.resolve()
    policy = args.policy.resolve()
    if not robot_xml.is_file():
        raise FileNotFoundError(robot_xml)
    if not policy.is_file():
        raise FileNotFoundError(policy)

    sim_module = load_upstream_sim(robot_xml)

    class LiveController(sim_module.RealTimePolicyController):
        def __init__(self, *controller_args: Any, **controller_kwargs: Any) -> None:
            super().__init__(*controller_args, **controller_kwargs)
            self.phase_clock = OfficialPhaseClock(args.policy_frequency, args.cycle_time)
            confirm_frames = max(1, round(args.fall_confirm_time * args.policy_frequency))
            self.fall_detector = LiveFallDetector(
                self.model,
                args.fall_orientation_deg,
                args.fall_root_height_min,
                confirm_frames,
            )
            self.last_reported_phase: str | None = None
            self.last_status_time = -math.inf

        def report_phase(
            self,
            phase: str,
            reasons: Sequence[str] = (),
            diagnostics: Mapping[str, Any] | None = None,
            force: bool = False,
        ) -> None:
            sim_time = float(self.data.time)
            changed = phase != self.last_reported_phase
            if (
                not changed
                and not force
                and sim_time - self.last_status_time < args.status_interval
            ):
                return
            self.last_status_time = sim_time
            self.last_reported_phase = phase
            details = diagnostics or {}
            detail = f" ({', '.join(reasons)})" if reasons else ""
            state = (
                f"tilt={details.get('tilt_deg', 0.0):.1f}deg "
                f"height={details.get('root_height', 0.0):.2f} "
                f"feet={'board' if details.get('feet_on_board') else 'off'} "
                f"confirm={details.get('confirm_frames', 0)} "
                f"clock={details.get('phase_value', 0.0):.3f}"
            )
            line = (
                f"[STATUS] t={sim_time:.2f}s phase={phase} "
                f"v={sim_module.v:.1f} h={sim_module.h:.2f} {state}{detail}"
            )
            if changed or force:
                print(f"\n[PHASE] {line}", flush=True)
            else:
                sys.stdout.write(f"\r{line.ljust(150)[:150]}")
                sys.stdout.flush()

        def reset_fall_state(self) -> None:
            self.fall_detector.reset()
            self.phase_clock.reset()
            self.last_reported_phase = None
            self.last_status_time = -math.inf

        def reset(self, init_pos: np.ndarray) -> None:
            super().reset(init_pos)
            self.reset_fall_state()
            _, _, diagnostics = self.fall_detector.check(self.data)
            diagnostics["phase_value"] = 0.0
            self.report_phase("push", diagnostics=diagnostics, force=True)

        def extract_data(self) -> Any:
            values = super().extract_data()
            phase, phase_value = self.phase_clock.next()
            fallen, reasons, diagnostics = self.fall_detector.check(self.data)
            if fallen:
                phase = "fall"
            diagnostics["phase_value"] = phase_value
            self.report_phase(phase, reasons, diagnostics)
            return values

    controller = LiveController(
        xml_file=str(robot_xml),
        policy_path=str(policy),
        device=args.device,
        policy_frequency=args.policy_frequency,
    )
    print(
        "[PHASE] official fixed schedule: "
        "push(0.0-0.4), push2steer(0.4-0.5), "
        "steer(0.5-0.95), steer2push(0.95-1.0); fall overrides"
    )
    controller.run()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the official HUSKY controller with real-time phase output.",
    )
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--key-events", type=Path)
    parser.add_argument("--key-map", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-id")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--robot-xml", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-frequency", type=int, default=50)
    parser.add_argument("--cycle-time", type=float, default=6.0)
    parser.add_argument("--fall-orientation-deg", type=float, default=70.0)
    parser.add_argument("--fall-root-height-min", type=float, default=0.45)
    parser.add_argument("--fall-confirm-time", type=float, default=0.2)
    parser.add_argument("--status-interval", type=float, default=0.1)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-neutral", action="store_true")
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--pre-padding", type=float, default=0.1)
    parser.add_argument("--post-padding", type=float, default=0.1)
    parser.add_argument("--merge-gap", type=float, default=0.1)
    parser.add_argument("--failure-margin", type=float, default=0.15)
    parser.add_argument("--post-reset-ignore", type=float, default=0.2)
    parser.add_argument("--root-height-min", type=float)
    parser.add_argument("--root-tilt-max-deg", type=float)
    args = parser.parse_args()
    if args.live:
        if args.robot_xml is None or args.policy is None:
            parser.error("--live requires --robot-xml and --policy")
        if args.policy_frequency <= 0:
            parser.error("--policy-frequency must be positive")
        if args.cycle_time <= 0:
            parser.error("--cycle-time must be positive")
        if args.fall_root_height_min <= 0:
            parser.error("--fall-root-height-min must be positive")
        if args.fall_confirm_time <= 0:
            parser.error("--fall-confirm-time must be positive")
        if args.status_interval <= 0:
            parser.error("--status-interval must be positive")
    else:
        missing = [
            option
            for option, value in (
                ("--rollout", args.rollout),
                ("--output-dir", args.output_dir),
                ("--episode-id", args.episode_id),
            )
            if value is None
        ]
        if missing:
            parser.error(f"split mode requires {', '.join(missing)}")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def flatten_mapping(payload: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        result[name] = value
        if isinstance(value, Mapping):
            result.update(flatten_mapping(value, name))
    return result


def field_by_name(payload: Any, names: Sequence[str]) -> tuple[str | None, Any]:
    flat = flatten_mapping(payload)
    for wanted in names:
        for name, value in flat.items():
            if name.split(".")[-1].lower() == wanted:
                return name, value
    return None, None


def unwrap_object_array(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value


def load_rollout(path: Path) -> Rollout:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        payload = unwrap_object_array(np.load(path, allow_pickle=True))
    elif suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            payload = {key: unwrap_object_array(archive[key]) for key in archive.files}
    elif suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        raise ValueError("rollout must use .npy, .npz, .pkl, or .pickle")

    keys = [str(key) for key in payload.keys()] if isinstance(payload, Mapping) else []
    shape = list(payload.shape) if isinstance(payload, np.ndarray) else None
    dtype = str(payload.dtype) if isinstance(payload, np.ndarray) else None
    num_frames = infer_num_frames(payload)
    return Rollout(path, suffix, payload, type(payload).__name__, keys, shape, dtype, num_frames)


def infer_num_frames(payload: Any) -> int:
    if isinstance(payload, np.ndarray):
        if payload.ndim == 0:
            raise ValueError("scalar ndarray does not define rollout frames")
        return int(payload.shape[0])
    if not isinstance(payload, Mapping):
        raise ValueError(f"unsupported rollout top-level type: {type(payload).__name__}")

    flat = flatten_mapping(payload)
    preferred: list[tuple[str, int]] = []
    for name, value in flat.items():
        value = np.asarray(value) if isinstance(value, (list, tuple)) else value
        if not isinstance(value, np.ndarray) or value.ndim == 0:
            continue
        leaf = name.split(".")[-1].lower()
        if leaf in {*TIME_FIELDS, *COMMAND_FIELDS, *QPOS_FIELDS, "frame_idx", "frame_index"}:
            preferred.append((name, int(value.shape[0])))
    if preferred:
        lengths = {length for _, length in preferred}
        if len(lengths) != 1:
            raise ValueError(f"conflicting frame counts in confirmed fields: {preferred}")
        return preferred[0][1]

    lengths = Counter()
    for value in flat.values():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            lengths[int(value.shape[0])] += 1
    if not lengths:
        raise ValueError("no frame-aligned arrays found")
    length, count = lengths.most_common(1)[0]
    if count < 2:
        raise ValueError(
            "frame count is ambiguous; provide a recognized time, command, qpos, "
            "or frame index field"
        )
    return length


def numeric_integrity(payload: Any, num_frames: int) -> dict[str, Any]:
    arrays: list[tuple[str, np.ndarray]] = []
    if isinstance(payload, np.ndarray):
        arrays.append(("<array>", payload))
    else:
        for name, value in flatten_mapping(payload).items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == num_frames:
                arrays.append((name, value))
    checked, nan_fields, inf_fields = [], [], []
    for name, value in arrays:
        if not np.issubdtype(value.dtype, np.number):
            continue
        checked.append(name)
        if np.isnan(value).any():
            nan_fields.append(name)
        if np.isinf(value).any():
            inf_fields.append(name)
    return {
        "checked_numeric_fields": checked,
        "has_nan": bool(nan_fields),
        "nan_fields": nan_fields,
        "has_inf": bool(inf_fields),
        "inf_fields": inf_fields,
    }


def video_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps", 0.0))
        frame_count = metadata.get("nframes")
        if not frame_count or not math.isfinite(float(frame_count)):
            frame_count = reader.count_frames()
        return {
            "path": str(path.resolve()),
            "fps": fps,
            "num_frames": int(frame_count),
            "duration": float(frame_count) / fps if fps > 0 else None,
        }
    finally:
        reader.close()


def infer_fps(
    rollout: Rollout, explicit_fps: float | None, video_info: Mapping[str, Any] | None
) -> tuple[float, str]:
    if explicit_fps is not None:
        if explicit_fps <= 0:
            raise ValueError("--fps must be positive")
        return float(explicit_fps), "command_line"

    _, time_value = field_by_name(rollout.payload, TIME_FIELDS)
    if time_value is not None:
        times = np.asarray(time_value, dtype=float).reshape(-1)
        if len(times) == rollout.num_frames:
            differences = np.diff(times)
            differences = differences[np.isfinite(differences) & (differences > 0)]
            if differences.size:
                return float(1.0 / np.median(differences)), "rollout_sim_time"

    _, scalar_fps = field_by_name(rollout.payload, ("fps", "control_fps", "policy_fps"))
    if scalar_fps is not None and np.asarray(scalar_fps).size == 1:
        fps = float(np.asarray(scalar_fps).reshape(-1)[0])
        if fps > 0:
            return fps, "rollout_scalar"

    if video_info and float(video_info["fps"]) > 0:
        return float(video_info["fps"]), "video_metadata"
    raise ValueError("unable to infer fps; pass --fps or provide simulation time/video")


def boolean_series(value: Any, num_frames: int) -> np.ndarray | None:
    array = np.asarray(value)
    if array.ndim == 0 or array.shape[0] != num_frames:
        return None
    if array.ndim > 1:
        array = np.any(array.astype(bool), axis=tuple(range(1, array.ndim)))
    return array.astype(bool)


def first_true(value: Any, num_frames: int) -> int | None:
    series = boolean_series(value, num_frames)
    if series is None:
        return None
    indices = np.flatnonzero(series)
    return int(indices[0]) if indices.size else None


def find_reset_frame(payload: Any, num_frames: int) -> tuple[int | None, str | None]:
    flat = flatten_mapping(payload)
    candidates = []
    for wanted in RESET_FIELDS:
        for name, value in flat.items():
            if name.split(".")[-1].lower() == wanted:
                frame = first_true(value, num_frames)
                if frame is not None:
                    candidates.append((frame, name))
    return min(candidates) if candidates else (None, None)


def quaternion_tilt_deg(quaternions: np.ndarray) -> np.ndarray:
    """Return root tilt from MuJoCo wxyz quaternions."""
    quat = np.asarray(quaternions, dtype=float)
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.where(norms > 0, norms, 1.0)
    w, x, y, z = (quat[:, index] for index in range(4))
    gravity_z = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(gravity_z, -1.0, 1.0)))


def detect_failure_frame(
    rollout: Rollout, root_height_min: float | None, root_tilt_max_deg: float | None
) -> tuple[int | None, str | None, str]:
    flat = flatten_mapping(rollout.payload)
    available_native = []
    detected = []
    for wanted in NATIVE_FAILURE_FIELDS:
        for name, value in flat.items():
            if name.split(".")[-1].lower() != wanted:
                continue
            series = boolean_series(value, rollout.num_frames)
            if series is None:
                continue
            available_native.append(name)
            indices = np.flatnonzero(series)
            if indices.size:
                detected.append((int(indices[0]), name))
    if detected:
        frame, reason = min(detected)
        return frame, reason, "native"
    if available_native:
        return None, None, "native_clear"

    fallback_checks = []
    if root_height_min is not None:
        name, value = field_by_name(rollout.payload, ROOT_POS_FIELDS)
        if value is None:
            raise RuntimeError(f"{UNKNOWN_FAILURE_DETECTION}: root position field missing")
        positions = np.asarray(value)
        if (
            positions.ndim != 2
            or positions.shape[0] != rollout.num_frames
            or positions.shape[1] < 3
        ):
            raise RuntimeError(f"{UNKNOWN_FAILURE_DETECTION}: invalid {name} shape")
        indices = np.flatnonzero(positions[:, 2] < root_height_min)
        if indices.size:
            fallback_checks.append((int(indices[0]), f"{name}<root_height_min"))

    if root_tilt_max_deg is not None:
        name, value = field_by_name(rollout.payload, ROOT_QUAT_FIELDS)
        if value is None:
            raise RuntimeError(f"{UNKNOWN_FAILURE_DETECTION}: root quaternion field missing")
        quaternions = np.asarray(value)
        if (
            quaternions.ndim != 2
            or quaternions.shape[0] != rollout.num_frames
            or quaternions.shape[1] != 4
        ):
            raise RuntimeError(f"{UNKNOWN_FAILURE_DETECTION}: invalid {name} shape")
        indices = np.flatnonzero(quaternion_tilt_deg(quaternions) > root_tilt_max_deg)
        if indices.size:
            fallback_checks.append((int(indices[0]), f"{name}>root_tilt_max_deg"))

    if root_height_min is None and root_tilt_max_deg is None:
        raise RuntimeError(
            f"{UNKNOWN_FAILURE_DETECTION}: no native failure field and no "
            "explicit fallback threshold"
        )
    if fallback_checks:
        frame, reason = min(fallback_checks)
        return frame, reason, "fallback"
    return None, None, "fallback_clear"


def normalize_key(key: str) -> str:
    value = str(key).strip().lower()
    aliases = {"key.space": "space", "key.enter": "enter", "key.return": "enter"}
    return aliases.get(value, value.strip("'\""))


def normalize_event_type(event_type: str) -> str:
    value = str(event_type).strip().lower().replace("-", "_")
    return {
        "down": "key_down",
        "press": "key_down",
        "pressed": "key_down",
        "up": "key_up",
        "release": "key_up",
        "released": "key_up",
    }.get(value, value)


def parse_optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def parse_key_event_rows(
    rows: Sequence[Mapping[str, Any]], fps: float, num_frames: int
) -> list[KeyEvent]:
    events = []
    for row_number, row in enumerate(rows, start=2):
        frame_value = row.get("frame_idx", row.get("frame_index"))
        sim_time = parse_optional_float(row.get("sim_time", row.get("time")))
        if frame_value is None or str(frame_value).strip() == "":
            if sim_time is None:
                raise ValueError(f"key event row {row_number} has no frame_idx or sim_time")
            frame_idx = int(round(sim_time * fps))
        else:
            frame_idx = int(frame_value)
        if not 0 <= frame_idx <= num_frames:
            raise ValueError(f"key event row {row_number} frame {frame_idx} is out of range")
        event_type = normalize_event_type(str(row.get("event_type", "")))
        if event_type not in {"key_down", "key_up", "command"}:
            raise ValueError(f"key event row {row_number} has invalid event_type: {event_type}")
        command = str(row.get("command", "")).strip() or None
        events.append(
            KeyEvent(
                frame_idx=frame_idx,
                sim_time=sim_time,
                key=normalize_key(str(row.get("key", ""))),
                event_type=event_type,
                command=command,
            )
        )
    return sorted(events, key=lambda event: event.frame_idx)


def load_key_events(path: Path, fps: float, num_frames: int) -> list[KeyEvent]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[Mapping[str, Any]]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        raise ValueError("key events must use .csv or .jsonl")
    return parse_key_event_rows(rows, fps, num_frames)


def key_events_from_rollout(
    rollout: Rollout, fps: float
) -> tuple[str | None, list[KeyEvent] | None]:
    name, value = field_by_name(
        rollout.payload, ("key_events", "keyboard_events", "keyboard_event")
    )
    if value is None:
        return None, None
    value = unwrap_object_array(value)
    rows: list[Mapping[str, Any]]
    if isinstance(value, Mapping):
        lengths = {
            len(column)
            for column in value.values()
            if isinstance(column, (list, tuple, np.ndarray))
        }
        if len(lengths) != 1:
            raise ValueError(f"{name} column lengths are inconsistent")
        length = lengths.pop()
        rows = [
            {
                key: column[index]
                for key, column in value.items()
                if isinstance(column, (list, tuple, np.ndarray))
            }
            for index in range(length)
        ]
    elif isinstance(value, np.ndarray) and value.dtype.names:
        rows = [{field: row[field].item() for field in value.dtype.names} for row in value]
    elif isinstance(value, np.ndarray) and value.dtype == object:
        rows = list(value.tolist())
    elif isinstance(value, (list, tuple)):
        rows = list(value)
    else:
        raise ValueError(f"{name} has unsupported type {type(value).__name__}")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{name} must contain event records")
    return name, parse_key_event_rows(rows, fps, rollout.num_frames)


def load_key_map(path: Path | None) -> tuple[dict[str, str], list[str]]:
    if path is None:
        return {}, []
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("key-map JSON must contain an object")
    if "mapping" in config:
        mapping_obj = config["mapping"]
        priority_obj = config.get("priority", [])
    else:
        mapping_obj = {key: value for key, value in config.items() if key != "priority"}
        priority_obj = config.get("priority", [])
    if not isinstance(mapping_obj, Mapping):
        raise ValueError("key-map 'mapping' must contain an object")
    mapping = {}
    for key, value in mapping_obj.items():
        if isinstance(value, Mapping):
            command = value.get("command")
        else:
            command = value
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"invalid command mapping for key {key!r}")
        mapping[normalize_key(str(key))] = command.strip()
    if not isinstance(priority_obj, list) or not all(
        isinstance(item, str) for item in priority_obj
    ):
        raise ValueError("key-map priority must be a list of keys")
    return mapping, [normalize_key(item) for item in priority_obj]


def event_rows(events: Sequence[KeyEvent]) -> list[dict[str, Any]]:
    return [
        {
            "frame_idx": event.frame_idx,
            "sim_time": "" if event.sim_time is None else event.sim_time,
            "key": event.key,
            "event_type": event.event_type,
            "command": event.command or "",
        }
        for event in events
    ]


def write_key_events(path: Path, events: Sequence[KeyEvent]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("frame_idx", "sim_time", "key", "event_type", "command")
        )
        writer.writeheader()
        writer.writerows(event_rows(events))


def active_command(
    active_keys: set[str], mapping: Mapping[str, str], priority: Sequence[str]
) -> tuple[str, str]:
    mapped = [(key, mapping[key]) for key in sorted(active_keys) if key in mapping]
    if not mapped:
        return "neutral", "+".join(sorted(active_keys))
    commands = {command for _, command in mapped}
    if len(commands) == 1:
        return mapped[0][1], "+".join(key for key, _ in mapped)
    if priority:
        for key in priority:
            if key in active_keys and key in mapping:
                return mapping[key], "+".join(sorted(active_keys))
    return "combined_command", "+".join(sorted(active_keys))


def command_history_from_events(
    events: Sequence[KeyEvent],
    mapping: Mapping[str, str],
    priority: Sequence[str],
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    commands = np.full(num_frames, "neutral", dtype=object)
    keys = np.full(num_frames, "", dtype=object)
    by_frame: dict[int, list[KeyEvent]] = defaultdict(list)
    for event in events:
        by_frame[event.frame_idx].append(event)

    active_keys: set[str] = set()
    direct_command: str | None = None
    for frame in range(num_frames):
        for event in by_frame.get(frame, []):
            if event.event_type == "key_down":
                if event.key:
                    active_keys.add(event.key)
            elif event.event_type == "key_up":
                active_keys.discard(event.key)
            if event.command:
                direct_command = event.command
            elif event.event_type == "command":
                raise ValueError(f"command event at frame {frame} has no command value")

        mapped_command, key = active_command(active_keys, mapping, priority)
        if mapping:
            command = mapped_command
        elif direct_command is not None:
            command = direct_command
        elif active_keys:
            raise ValueError("key events have no commands and no --key-map was supplied")
        else:
            command = "neutral"
        commands[frame] = command
        keys[frame] = key
    return commands, keys


def command_history_from_rollout(rollout: Rollout) -> tuple[np.ndarray, np.ndarray]:
    name, value = field_by_name(rollout.payload, COMMAND_FIELDS)
    if value is None:
        raise ValueError("no --key-events and no rollout command history were found")
    commands = np.asarray(value).reshape(-1)
    if len(commands) != rollout.num_frames:
        raise ValueError(f"{name} is not frame-aligned")
    if commands.dtype.kind not in {"U", "S", "O"}:
        raise ValueError(
            f"{name} is numeric; semantic segmentation requires key events or string commands"
        )
    values = np.asarray(
        [item.decode() if isinstance(item, (bytes, np.bytes_)) else str(item) for item in commands],
        dtype=object,
    )
    return values, np.full(rollout.num_frames, "", dtype=object)


def runs(commands: np.ndarray, keys: np.ndarray) -> list[Span]:
    if not len(commands):
        return []
    spans = []
    start = 0
    for frame in range(1, len(commands) + 1):
        if frame == len(commands) or commands[frame] != commands[start]:
            key_values = [str(item) for item in keys[start:frame] if str(item)]
            key = Counter(key_values).most_common(1)[0][0] if key_values else ""
            spans.append(Span(str(commands[start]), key, start, frame))
            start = frame
    return spans


def merge_spans(spans: Sequence[Span], max_gap_frames: int) -> list[Span]:
    if max_gap_frames <= 0:
        return list(spans)
    merged: list[Span] = []
    index = 0
    while index < len(spans):
        current = spans[index]
        if (
            index + 2 < len(spans)
            and current.command != "neutral"
            and spans[index + 1].command == "neutral"
            and spans[index + 1].end - spans[index + 1].start <= max_gap_frames
            and spans[index + 2].command == current.command
        ):
            next_span = spans[index + 2]
            merged.append(
                Span(
                    current.command,
                    current.key or next_span.key,
                    current.start,
                    next_span.end,
                )
            )
            index += 3
            continue
        merged.append(current)
        index += 1
    return merge_spans(merged, max_gap_frames) if len(merged) < len(spans) else merged


def safe_label(command: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_-]+", "_", command.strip().lower()).strip("_")
    return label or "unlabelled"


def build_segments(
    spans: Sequence[Span],
    episode_id: str,
    fps: float,
    num_frames: int,
    failure_frame: int | None,
    failure_reason: str | None,
    valid_end_frame: int,
    reset_frame: int | None,
    min_duration: float,
    pre_padding: float,
    post_padding: float,
    include_neutral: bool,
) -> list[Segment]:
    pre_frames = round(pre_padding * fps)
    post_frames = round(post_padding * fps)
    minimum_frames = math.ceil(min_duration * fps)
    counts: Counter[str] = Counter()
    segments = []

    for span in spans:
        label = safe_label(span.command)
        identifier = f"{label}_{counts[label]:03d}"
        counts[label] += 1
        original_end = span.end
        start = max(0, span.start - pre_frames)
        end = min(num_frames, span.end + post_frames)
        truncated = failure_frame is not None and end > valid_end_frame and start < valid_end_frame
        status = "valid"
        discard_reason = ""

        if reset_frame is not None and start >= reset_frame:
            status, discard_reason = "discarded", "after_reset"
        elif failure_frame is not None and start >= failure_frame:
            status, discard_reason = "discarded", "after_failure"
        elif start >= valid_end_frame:
            status, discard_reason = "discarded", "failure_margin"
        else:
            end = min(end, valid_end_frame)

        if span.command == "neutral" and not include_neutral:
            status, discard_reason = "discarded", "neutral_excluded"
        if end <= start:
            status, discard_reason = "discarded", discard_reason or "empty_after_clipping"
        elif end - start < minimum_frames:
            status, discard_reason = "discarded", discard_reason or "shorter_than_min_duration"

        num_segment_frames = max(0, end - start)
        segments.append(
            Segment(
                segment_id=identifier,
                episode_id=episode_id,
                motion_label=label,
                key=span.key,
                command=span.command,
                start_frame=start,
                end_frame=end,
                start_time=start / fps,
                end_time=end / fps,
                duration=num_segment_frames / fps,
                num_frames=num_segment_frames,
                status=status,
                failure_detected=failure_frame is not None,
                failure_frame=failure_frame,
                failure_reason=failure_reason or "",
                truncated_by_failure=truncated,
                original_end_frame=original_end,
                valid_end_frame=valid_end_frame,
                reset_detected=reset_frame is not None,
                discard_reason=discard_reason,
                notes="end_frame is exclusive",
            )
        )
    return segments


def slice_payload(value: Any, start: int, end: int, num_frames: int) -> Any:
    if isinstance(value, np.ndarray):
        return value[start:end] if value.ndim > 0 and value.shape[0] == num_frames else value
    if isinstance(value, Mapping):
        return {key: slice_payload(item, start, end, num_frames) for key, item in value.items()}
    if isinstance(value, list) and len(value) == num_frames:
        return value[start:end]
    if isinstance(value, tuple) and len(value) == num_frames:
        return value[start:end]
    return value


def save_segment_payload(rollout: Rollout, segment: Segment, destination_stem: Path) -> Path:
    sliced = slice_payload(
        rollout.payload, segment.start_frame, segment.end_frame, rollout.num_frames
    )
    if rollout.suffix == ".npy":
        path = destination_stem.with_suffix(".npy")
        np.save(path, sliced, allow_pickle=True)
    elif rollout.suffix == ".npz":
        path = destination_stem.with_suffix(".npz")
        if not isinstance(sliced, Mapping):
            raise ValueError("NPZ rollout did not load as a mapping")
        np.savez(path, **sliced)
    else:
        path = destination_stem.with_suffix(".pkl")
        with path.open("wb") as handle:
            pickle.dump(sliced, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def write_segments_csv(path: Path, segments: Sequence[Segment]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEGMENT_FIELDS)
        writer.writeheader()
        for segment in segments:
            writer.writerow(asdict(segment))


def relative_to_episode(path: Path, episode_dir: Path) -> str:
    return str(path.relative_to(episode_dir))


def validate_segments(segments: Sequence[Segment], num_frames: int, fps: float) -> None:
    for segment in segments:
        if segment.status != "valid":
            continue
        if not (0 <= segment.start_frame < segment.end_frame <= num_frames):
            raise AssertionError(f"invalid frame range for {segment.segment_id}")
        if segment.num_frames != segment.end_frame - segment.start_frame:
            raise AssertionError(f"frame count mismatch for {segment.segment_id}")
        if not math.isclose(segment.duration, segment.num_frames / fps, abs_tol=1e-9):
            raise AssertionError(f"duration mismatch for {segment.segment_id}")
        if segment.failure_frame is not None and segment.end_frame > segment.valid_end_frame:
            raise AssertionError(f"failure boundary crossed by {segment.segment_id}")


def overlay_frame(frame: np.ndarray, lines: Iterable[str], invalid: bool = False) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(np.asarray(frame).astype(np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    text_lines = list(lines)
    line_height = 14
    width = min(image.width, max(280, max((len(line) for line in text_lines), default=1) * 7))
    draw.rectangle((0, 0, width, 10 + line_height * len(text_lines)), fill=(0, 0, 0, 175))
    for index, line in enumerate(text_lines):
        draw.text((6, 5 + index * line_height), line, fill=(255, 255, 255, 255), font=font)
    if invalid:
        draw.rectangle((0, image.height - 30, image.width, image.height), fill=(150, 0, 0, 190))
        draw.text((8, image.height - 23), "INVALID: failure/reset region", fill="white", font=font)
    return np.asarray(image)


def segment_at_frame(segments: Sequence[Segment], frame: int) -> Segment | None:
    for segment in segments:
        if segment.status == "valid" and segment.start_frame <= frame < segment.end_frame:
            return segment
    return None


def render_from_synchronized_video(
    video_path: Path,
    video_info: Mapping[str, Any],
    episode_dir: Path,
    episode_id: str,
    segments: Sequence[Segment],
    commands: np.ndarray,
    keys: np.ndarray,
    rollout_fps: float,
    num_frames: int,
    failure_frame: int | None,
    valid_end_frame: int,
    reset_frame: int | None,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    video_fps = float(video_info["fps"])
    video_frames = int(video_info["num_frames"])
    rollout_duration = num_frames / rollout_fps
    video_duration = video_frames / video_fps
    tolerance = max(0.25, 2.0 / rollout_fps)
    if abs(rollout_duration - video_duration) > tolerance:
        raise ValueError(
            "video/rollout synchronization check failed: "
            f"{video_duration:.3f}s vs {rollout_duration:.3f}s"
        )

    preview_dir = episode_dir / "preview"
    preview_dir.mkdir(exist_ok=True)
    preview_path = preview_dir / "full_rollout_with_segments.mp4"
    preview_writer = imageio.get_writer(preview_path, fps=video_fps, macro_block_size=1)
    writers: dict[str, Any] = {}
    segment_paths: dict[str, Path] = {}
    for segment in segments:
        if segment.status != "valid":
            continue
        path = episode_dir / segment.motion_label / f"{segment.segment_id}.mp4"
        writers[segment.segment_id] = imageio.get_writer(path, fps=video_fps, macro_block_size=1)
        segment_paths[segment.segment_id] = path

    failure_writer = None
    failure_path = None
    failure_video_start = None
    if failure_frame is not None:
        failure_dir = episode_dir / "failures"
        failure_dir.mkdir(exist_ok=True)
        failure_path = failure_dir / "failure_preview.mp4"
        failure_writer = imageio.get_writer(failure_path, fps=video_fps, macro_block_size=1)
        failure_video_start = max(0, failure_frame - round(rollout_fps))

    reader = imageio.get_reader(video_path)
    decoded = 0
    try:
        for video_index, frame in enumerate(reader):
            decoded += 1
            rollout_frame = min(num_frames - 1, int(video_index * rollout_fps / video_fps))
            segment = segment_at_frame(segments, rollout_frame)
            markers = []
            if rollout_frame == failure_frame:
                markers.append("FAILURE")
            if rollout_frame == valid_end_frame:
                markers.append("VALID_END")
            if rollout_frame == reset_frame:
                markers.append("RESET")
            lines = [
                f"episode: {episode_id}",
                f"frame/time: {rollout_frame} / {rollout_frame / rollout_fps:.3f}s",
                f"key: {keys[rollout_frame] or '-'}",
                f"command: {commands[rollout_frame]}",
                f"segment: {segment.segment_id if segment else '-'}",
                f"boundary: {', '.join(markers) if markers else '-'}",
            ]
            invalid = rollout_frame >= valid_end_frame
            preview_writer.append_data(overlay_frame(frame, lines, invalid))

            if segment is not None:
                segment_lines = [
                    f"motion: {segment.motion_label}",
                    f"key/command: {segment.key or '-'} / {segment.command}",
                    f"episode: {episode_id}",
                    f"time: {segment.start_time:.3f}-{segment.end_time:.3f}s",
                ]
                if segment.truncated_by_failure:
                    segment_lines.append("truncated_before_failure")
                writers[segment.segment_id].append_data(overlay_frame(frame, segment_lines))

            if (
                failure_writer is not None
                and failure_video_start is not None
                and rollout_frame >= failure_video_start
            ):
                failure_writer.append_data(overlay_frame(frame, lines, invalid))
    finally:
        reader.close()
        preview_writer.close()
        for writer in writers.values():
            writer.close()
        if failure_writer is not None:
            failure_writer.close()

    for segment in segments:
        if segment.segment_id in segment_paths:
            segment.video_path = relative_to_episode(segment_paths[segment.segment_id], episode_dir)
    return {
        "method": "synchronized_video",
        "source_video": str(video_path.resolve()),
        "video_fps": video_fps,
        "decoded_frames": decoded,
        "preview_path": relative_to_episode(preview_path, episode_dir),
        "failure_preview_path": (
            relative_to_episode(failure_path, episode_dir) if failure_path else None
        ),
    }


def find_qpos(rollout: Rollout) -> tuple[str | None, np.ndarray | None]:
    name, value = field_by_name(rollout.payload, QPOS_FIELDS)
    if value is None:
        return None, None
    qpos = np.asarray(value)
    if qpos.ndim != 2 or qpos.shape[0] != rollout.num_frames:
        return name, None
    return name, qpos


def render_from_qpos(
    robot_xml: Path,
    rollout: Rollout,
    episode_dir: Path,
    episode_id: str,
    segments: Sequence[Segment],
    commands: np.ndarray,
    keys: np.ndarray,
    fps: float,
    failure_frame: int | None,
    valid_end_frame: int,
    reset_frame: int | None,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import mujoco

    qpos_name, qpos = find_qpos(rollout)
    if qpos is None:
        raise ValueError("pose replay requires a frame-aligned qpos field")
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"qpos width {qpos.shape[1]} != model.nq {model.nq}; joint mapping is UNKNOWN"
        )

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.0
    camera.azimuth = 135.0
    camera.elevation = -20.0

    preview_dir = episode_dir / "preview"
    preview_dir.mkdir(exist_ok=True)
    preview_path = preview_dir / "full_rollout_with_segments.mp4"
    preview_writer = imageio.get_writer(preview_path, fps=fps, macro_block_size=1)
    writers: dict[str, Any] = {}
    segment_paths: dict[str, Path] = {}
    for segment in segments:
        if segment.status == "valid":
            path = episode_dir / segment.motion_label / f"{segment.segment_id}.mp4"
            writers[segment.segment_id] = imageio.get_writer(path, fps=fps, macro_block_size=1)
            segment_paths[segment.segment_id] = path

    failure_writer = None
    failure_path = None
    failure_start = None
    if failure_frame is not None:
        failure_dir = episode_dir / "failures"
        failure_dir.mkdir(exist_ok=True)
        failure_path = failure_dir / "failure_preview.mp4"
        failure_writer = imageio.get_writer(failure_path, fps=fps, macro_block_size=1)
        failure_start = max(0, failure_frame - round(fps))

    try:
        for frame_index in range(rollout.num_frames):
            data.qpos[:] = qpos[frame_index]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            segment = segment_at_frame(segments, frame_index)
            markers = []
            if frame_index == failure_frame:
                markers.append("FAILURE")
            if frame_index == valid_end_frame:
                markers.append("VALID_END")
            if frame_index == reset_frame:
                markers.append("RESET")
            lines = [
                f"episode: {episode_id}",
                f"frame/time: {frame_index} / {frame_index / fps:.3f}s",
                f"key: {keys[frame_index] or '-'}",
                f"command: {commands[frame_index]}",
                f"segment: {segment.segment_id if segment else '-'}",
                f"boundary: {', '.join(markers) if markers else '-'}",
            ]
            invalid = frame_index >= valid_end_frame
            preview_writer.append_data(overlay_frame(frame, lines, invalid))
            if segment is not None:
                segment_lines = [
                    f"motion: {segment.motion_label}",
                    f"key/command: {segment.key or '-'} / {segment.command}",
                    f"episode: {episode_id}",
                    f"time: {segment.start_time:.3f}-{segment.end_time:.3f}s",
                ]
                if segment.truncated_by_failure:
                    segment_lines.append("truncated_before_failure")
                writers[segment.segment_id].append_data(overlay_frame(frame, segment_lines))
            if (
                failure_writer is not None
                and failure_start is not None
                and frame_index >= failure_start
            ):
                failure_writer.append_data(overlay_frame(frame, lines, invalid))
    finally:
        renderer.close()
        preview_writer.close()
        for writer in writers.values():
            writer.close()
        if failure_writer is not None:
            failure_writer.close()

    for segment in segments:
        if segment.segment_id in segment_paths:
            segment.video_path = relative_to_episode(segment_paths[segment.segment_id], episode_dir)
    return {
        "method": "mujoco_qpos_replay",
        "robot_xml": str(robot_xml.resolve()),
        "qpos_field": qpos_name,
        "model_nq": model.nq,
        "quaternion_convention": "MuJoCo wxyz",
        "preview_path": relative_to_episode(preview_path, episode_dir),
        "failure_preview_path": (
            relative_to_episode(failure_path, episode_dir) if failure_path else None
        ),
    }


def rollout_summary(
    rollout: Rollout,
    fps: float,
    fps_source: str,
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    flat = flatten_mapping(rollout.payload)
    fields = {}
    for name, value in flat.items():
        if isinstance(value, np.ndarray):
            fields[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif not isinstance(value, Mapping):
            fields[name] = {"type": type(value).__name__}
    return {
        "file_type": rollout.suffix,
        "top_level_type": rollout.top_type,
        "top_level_keys": rollout.keys,
        "shape": rollout.shape,
        "dtype": rollout.dtype,
        "num_frames": rollout.num_frames,
        "fps": fps,
        "fps_source": fps_source,
        "fields": fields,
        "numeric_integrity": dict(integrity),
    }


def prepare_output(
    output_dir: Path,
    episode_id: str,
    overwrite: bool,
    protected_paths: Sequence[Path | None],
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", episode_id):
        raise ValueError("--episode-id may contain only letters, numbers, underscores, and hyphens")
    episode_dir = output_dir / episode_id
    resolved_episode_dir = episode_dir.resolve()
    for protected_path in protected_paths:
        if protected_path is None:
            continue
        try:
            protected_path.resolve().relative_to(resolved_episode_dir)
        except ValueError:
            continue
        raise ValueError(f"refusing output directory containing source input: {protected_path}")
    if episode_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{episode_dir} already exists; pass --overwrite to replace this test output"
            )
        shutil.rmtree(episode_dir)
    episode_dir.mkdir(parents=True)
    return episode_dir


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    if args.live:
        return run_live(args)
    source_hash_before = sha256(args.rollout)
    rollout = load_rollout(args.rollout)
    video_info = video_metadata(args.video)
    fps, fps_source = infer_fps(rollout, args.fps, video_info)
    integrity = numeric_integrity(rollout.payload, rollout.num_frames)
    schema = rollout_summary(rollout, fps, fps_source, integrity)
    print(json.dumps(schema, indent=2))

    episode_dir = prepare_output(
        args.output_dir,
        args.episode_id,
        args.overwrite,
        (args.rollout, args.key_events, args.key_map, args.video, args.robot_xml),
    )
    raw_reference = {
        "episode_id": args.episode_id,
        "source_rollout_path": str(args.rollout.resolve()),
        "source_rollout_sha256": source_hash_before,
        "source_rollout_size_bytes": args.rollout.stat().st_size,
        "source_video": str(args.video.resolve()) if args.video else None,
        "source_key_events": str(args.key_events.resolve()) if args.key_events else None,
        "source_key_map": str(args.key_map.resolve()) if args.key_map else None,
        "rollout_schema": schema,
    }
    write_json(episode_dir / "raw_reference.json", raw_reference)

    report: dict[str, Any] = {
        "status": "processing",
        "episode_id": args.episode_id,
        "rollout_schema": schema,
        "video_input": video_info,
        "unknown_fields": [],
    }
    try:
        failure_frame, failure_reason, failure_source = detect_failure_frame(
            rollout, args.root_height_min, args.root_tilt_max_deg
        )
        reset_frame, reset_reason = find_reset_frame(rollout.payload, rollout.num_frames)
        margin_frames = round(args.failure_margin * fps)
        valid_end_frame = rollout.num_frames
        if failure_frame is not None:
            valid_end_frame = max(0, failure_frame - margin_frames)
        if reset_frame is not None:
            valid_end_frame = min(valid_end_frame, reset_frame)

        embedded_event_field = None
        if args.key_events:
            events = load_key_events(args.key_events, fps, rollout.num_frames)
            mapping, priority = load_key_map(args.key_map)
            commands, keys = command_history_from_events(
                events, mapping, priority, rollout.num_frames
            )
            write_key_events(episode_dir / "key_events.csv", events)
            command_source = "key_events"
        else:
            embedded_event_field, embedded_events = key_events_from_rollout(rollout, fps)
            mapping, priority = load_key_map(args.key_map)
            if embedded_events is not None:
                events = embedded_events
                commands, keys = command_history_from_events(
                    events, mapping, priority, rollout.num_frames
                )
                command_source = f"rollout:{embedded_event_field}"
            else:
                events = []
                commands, keys = command_history_from_rollout(rollout)
                command_source = "rollout_command_history"
            write_key_events(episode_dir / "key_events.csv", events)

        spans = merge_spans(runs(commands, keys), round(args.merge_gap * fps))
        segments = build_segments(
            spans=spans,
            episode_id=args.episode_id,
            fps=fps,
            num_frames=rollout.num_frames,
            failure_frame=failure_frame,
            failure_reason=failure_reason,
            valid_end_frame=valid_end_frame,
            reset_frame=reset_frame,
            min_duration=args.min_duration,
            pre_padding=args.pre_padding,
            post_padding=args.post_padding,
            include_neutral=args.include_neutral,
        )
        validate_segments(segments, rollout.num_frames, fps)

        for segment in segments:
            if segment.status != "valid":
                continue
            directory = episode_dir / segment.motion_label
            directory.mkdir(exist_ok=True)
            pose_path = save_segment_payload(rollout, segment, directory / segment.segment_id)
            segment.pose_path = relative_to_episode(pose_path, episode_dir)
            metadata = asdict(segment)
            metadata.update(
                {
                    "source_rollout_path": str(args.rollout.resolve()),
                    "source_rollout_sha256": source_hash_before,
                    "source_format": rollout.suffix,
                    "sliced_existing_fields_only": True,
                }
            )
            write_json(directory / f"{segment.segment_id}.json", metadata)

        video_result: dict[str, Any]
        try:
            if args.video:
                assert video_info is not None
                video_result = render_from_synchronized_video(
                    args.video,
                    video_info,
                    episode_dir,
                    args.episode_id,
                    segments,
                    commands,
                    keys,
                    fps,
                    rollout.num_frames,
                    failure_frame,
                    valid_end_frame,
                    reset_frame,
                )
            elif args.robot_xml:
                video_result = render_from_qpos(
                    args.robot_xml,
                    rollout,
                    episode_dir,
                    args.episode_id,
                    segments,
                    commands,
                    keys,
                    fps,
                    failure_frame,
                    valid_end_frame,
                    reset_frame,
                )
            else:
                qpos_name, qpos = find_qpos(rollout)
                missing = ["--video"]
                if qpos is None:
                    missing.append("frame-aligned qpos")
                missing.append("--robot-xml")
                video_result = {
                    "method": "unavailable",
                    "missing": missing,
                    "qpos_field": qpos_name,
                    "note": "pose segments retained; no fake video generated",
                }
                report["unknown_fields"].append("video_generation_capability")
        except Exception as error:
            video_result = {
                "method": "failed",
                "error": f"{type(error).__name__}: {error}",
                "note": "pose segments retained; no fake video generated",
            }
            report["unknown_fields"].append("video_generation_failed")

        if failure_frame is not None:
            failure_dir = episode_dir / "failures"
            failure_dir.mkdir(exist_ok=True)
            write_json(
                failure_dir / "failure_info.json",
                {
                    "failure_frame": failure_frame,
                    "failure_time": failure_frame / fps,
                    "failure_reason": failure_reason,
                    "failure_source": failure_source,
                    "failure_margin_frames": margin_frames,
                    "valid_end_frame": valid_end_frame,
                    "valid_end_time": valid_end_frame / fps,
                },
            )

        source_hash_after = sha256(args.rollout)
        if source_hash_after != source_hash_before:
            raise RuntimeError("source rollout changed during processing")
        write_segments_csv(episode_dir / "segments.csv", segments)

        report.update(
            {
                "status": "completed",
                "command_source": command_source,
                "key_mapping": mapping,
                "key_priority": priority or "UNKNOWN",
                "failure_detected": failure_frame is not None,
                "failure_frame": failure_frame,
                "failure_reason": failure_reason,
                "failure_detection_source": failure_source,
                "failure_margin_frames": margin_frames,
                "valid_end_frame": valid_end_frame,
                "reset_detected": reset_frame is not None,
                "reset_frame": reset_frame,
                "reset_reason": reset_reason,
                "post_reset_ignore_frames": round(args.post_reset_ignore * fps),
                "first_reset_ends_processing": True,
                "segments": [asdict(segment) for segment in segments],
                "valid_segment_count": sum(segment.status == "valid" for segment in segments),
                "discarded_segment_count": sum(segment.status != "valid" for segment in segments),
                "video": video_result,
                "source_sha256_before": source_hash_before,
                "source_sha256_after": source_hash_after,
                "source_unchanged": True,
            }
        )
        write_json(episode_dir / "processing_report.json", report)
        # Rewrite after video paths have been assigned.
        write_segments_csv(episode_dir / "segments.csv", segments)
        for segment in segments:
            if segment.status == "valid":
                metadata_path = episode_dir / segment.motion_label / f"{segment.segment_id}.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata.update(asdict(segment))
                write_json(metadata_path, metadata)
        print(f"Output: {episode_dir}")
        print(
            f"Valid segments: {report['valid_segment_count']}; "
            f"discarded: {report['discarded_segment_count']}"
        )
        if video_result["method"] in {"unavailable", "failed"}:
            video_detail = video_result.get("missing") or video_result.get("error")
            print(f"Video: {video_result['method']} ({video_detail})")
        return 0
    except RuntimeError as error:
        if UNKNOWN_FAILURE_DETECTION not in str(error):
            raise
        report.update(
            {
                "status": UNKNOWN_FAILURE_DETECTION,
                "error": str(error),
                "export_stopped": True,
                "source_unchanged": sha256(args.rollout) == source_hash_before,
            }
        )
        write_json(episode_dir / "processing_report.json", report)
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
