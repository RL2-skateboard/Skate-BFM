from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

from skate_bfm.exp.h1_bfm_coverage.core import (
    EXPERIMENT_TYPE,
    CheckpointCompatibilityError,
    ExpertTarget,
    H1RolloutRunner,
    ScoredRollout,
    angular_distance,
    classify_coverage,
    evaluate_geodesic_support,
    evaluate_robustness,
    load_bfm0_checkpoint,
    load_expert_targets,
    run_cem,
    sample_global_latents,
    score_rollout,
    spherical_lerp,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


class _JsonEncoder(json.JSONEncoder):
    def default(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return super().default(value)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, cls=_JsonEncoder) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-type", choices=("smoke", "formal"))
    parser.add_argument("--experiment-name")
    parser.add_argument("--device")
    video = parser.add_mutually_exclusive_group()
    video.add_argument("--save-video", dest="save_video", action="store_true")
    video.add_argument("--no-save-video", dest="save_video", action="store_false")
    parser.set_defaults(save_video=None)
    return parser.parse_args(argv)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    path = _resolve_path(args.config)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("H1 config must contain a YAML mapping")
    config = copy.deepcopy(config)
    config["checkpoint"]["path"] = str(args.checkpoint)
    if args.run_type:
        config["experiment"]["run_type"] = args.run_type
    if args.device:
        config["checkpoint"]["device"] = args.device
    if args.save_video is not None:
        config["visualization"]["save_video"] = args.save_video
    if config["experiment"]["run_type"] == "smoke":
        config["rollout"]["horizon_seconds"] = 0.04
        config["rollout"]["seeds"] = [0]
        config["expert_data"]["enable_push_pose"] = True
        config["expert_data"]["enable_steer_pose"] = False
        config["expert_data"]["enable_human_push"] = False
        config["global_scan"]["num_latents"] = 4
        config["cem"]["population_size"] = 4
        config["cem"]["num_iterations"] = 1
        config["geodesic"]["angles_degrees"] = [10]
        config["geodesic"]["samples_per_angle"] = 2
        config["robustness"]["trials"] = 2
    return config


def _git_metadata() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit or "unknown", dirty


def _setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"skate_bfm.h1.{path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.handlers[:] = [file_handler, stream_handler]
    return logger


def _run_name(config: dict[str, Any], requested: str | None, start: datetime) -> str:
    if requested:
        return requested
    return f"{config['experiment']['base_name']}_{start.strftime('%Y%m%d_%H%M%S')}"


def _device(config: dict[str, Any]) -> str:
    requested = str(config["checkpoint"]["device"])
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if config["experiment"]["run_type"] == "formal":
            raise RuntimeError(f"Formal run requested unavailable device: {requested}")
        return "cpu"
    return requested


def _latent_record(
    latent: torch.Tensor | np.ndarray,
    source: str,
    target: str,
    result: ScoredRollout,
    *,
    angle_degrees: float = math.nan,
    iteration: int = -1,
    fraction: float = math.nan,
    search_seed: int = -1,
) -> dict[str, Any]:
    return {
        "latent": np.asarray(torch.as_tensor(latent).detach().cpu(), dtype=np.float32),
        "source": source,
        "target": target,
        "score": result.score,
        "success": result.success,
        "fall": result.rollout.fall,
        "angle_degrees": angle_degrees,
        "iteration": iteration,
        "fraction": fraction,
        "search_seed": search_seed,
    }


def _trajectory_arrays(result: ScoredRollout) -> dict[str, np.ndarray]:
    states = result.rollout.states
    return {
        "root_position": np.stack([state["root_position"] for state in states]),
        "root_quaternion": np.stack([state["root_quaternion"] for state in states]),
        "joint_position": np.stack([state["joint_position"] for state in states]),
        "joint_velocity": np.stack([state["joint_velocity"] for state in states]),
        "board_position": np.stack([state["board_position"] for state in states]),
        "board_quaternion": np.stack([state["board_quaternion"] for state in states]),
        "actions": result.rollout.actions,
    }


def _safe_key(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _save_latents(path: Path, records: list[dict[str, Any]]) -> None:
    np.savez_compressed(
        path,
        latents=np.stack([record["latent"] for record in records]),
        source=np.asarray([record["source"] for record in records]),
        target=np.asarray([record["target"] for record in records]),
        score=np.asarray([record["score"] for record in records], dtype=np.float64),
        success=np.asarray([record["success"] for record in records], dtype=bool),
        fall=np.asarray([record["fall"] for record in records], dtype=bool),
        angle_degrees=np.asarray([record["angle_degrees"] for record in records], dtype=np.float64),
        iteration=np.asarray([record["iteration"] for record in records], dtype=int),
        fraction=np.asarray([record["fraction"] for record in records], dtype=np.float64),
        search_seed=np.asarray([record["search_seed"] for record in records], dtype=int),
    )


def _save_trajectories(
    path: Path,
    trajectories: dict[str, ScoredRollout],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for name, result in trajectories.items():
        for field, value in _trajectory_arrays(result).items():
            arrays[f"{_safe_key(name)}__{field}"] = value
    np.savez_compressed(path, **arrays)


def _write_target_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "target",
        "kind",
        "encoded_anchor_available",
        "global_best_score",
        "global_success_rate",
        "searched_anchor_score",
        "searched_success",
        "robust_success_rate",
        "search_angle_degrees",
        "angular_support",
        "coverage_type",
        "global_best_metrics",
        "searched_anchor_metrics",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["global_best_metrics"] = json.dumps(row["global_best_metrics"])
            output["searched_anchor_metrics"] = json.dumps(row["searched_anchor_metrics"])
            writer.writerow(output)


def _embedding(records: list[dict[str, Any]], dimensions: int, random_state: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    values = np.stack([record["latent"] for record in records]).astype(np.float64)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    if len(values) <= dimensions:
        padded = np.zeros((len(values), dimensions))
        padded[:, : min(values.shape[1], dimensions)] = values[:, :dimensions]
        return padded
    perplexity = min(30.0, max(2.0, (len(values) - 1) / 3.0))
    perplexity = min(perplexity, len(values) - 1.0)
    return TSNE(
        n_components=dimensions,
        metric="cosine",
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate="auto",
    ).fit_transform(values)


def _record_color(record: dict[str, Any]) -> str:
    source = record["source"]
    target = record["target"]
    if source == "global":
        return "#8a8f98"
    if source == "slerp":
        return "#7c3aed"
    if source == "cem":
        return "#ea7c23"
    if source == "human_push_anchor":
        return "#2f6fed"
    if source == "searched_anchor":
        return "#c83f49" if "push" in target else "#23875c"
    if source == "geodesic":
        return "#5b8def" if "push" in target else "#58a67a"
    if "push" in target:
        return "#c83f49"
    return "#23875c"


def _plot_latent_tsne(
    records: list[dict[str, Any]],
    path: Path,
    random_state: int,
) -> None:
    import matplotlib.pyplot as plt

    coordinates = _embedding(records, 2, random_state)
    figure, axis = plt.subplots(figsize=(8, 6))
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record["source"], []).append(index)
    for source, indices in groups.items():
        axis.scatter(
            coordinates[indices, 0],
            coordinates[indices, 1],
            c=[_record_color(records[index]) for index in indices],
            s=20 if source == "global" else 42,
            alpha=0.75,
            label=source,
        )
    for target in {record["target"] for record in records}:
        for source in ("cem", "slerp"):
            search_seeds = (
                {
                    record["search_seed"]
                    for record in records
                    if record["target"] == target and record["source"] == source
                }
                if source == "cem"
                else {-1}
            )
            for search_seed in search_seeds:
                indices = [
                    index
                    for index, record in enumerate(records)
                    if record["target"] == target
                    and record["source"] == source
                    and (source != "cem" or record["search_seed"] == search_seed)
                ]
                if len(indices) > 1:
                    order_key = "iteration" if source == "cem" else "fraction"
                    indices.sort(key=lambda index: records[index][order_key])
                    axis.plot(
                        coordinates[indices, 0],
                        coordinates[indices, 1],
                        color=_record_color(records[indices[0]]),
                        linewidth=1.2,
                        alpha=0.8,
                    )
    human_push_indices = [
        index for index, record in enumerate(records) if record["source"] == "human_push_anchor"
    ]
    if len(human_push_indices) > 1:
        human_push_indices.sort(key=lambda index: records[index]["target"])
        axis.plot(
            coordinates[human_push_indices, 0],
            coordinates[human_push_indices, 1],
            color="#2f6fed",
            linewidth=1.2,
            alpha=0.8,
        )
    axis.set_title("BFM0 latent directions: cosine t-SNE")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_latent_sphere(
    records: list[dict[str, Any]],
    path: Path,
    random_state: int,
) -> None:
    import matplotlib.pyplot as plt

    coordinates = _embedding(records, 3, random_state)
    norms = np.linalg.norm(coordinates, axis=1, keepdims=True)
    coordinates = coordinates / np.maximum(norms, 1e-12)
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    axis.plot_surface(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color="#cfd4db",
        alpha=0.08,
        linewidth=0,
    )
    for source in sorted({record["source"] for record in records}):
        indices = [index for index, record in enumerate(records) if record["source"] == source]
        sizes = [18 + 22 * records[index]["success"] for index in indices]
        axis.scatter(
            coordinates[indices, 0],
            coordinates[indices, 1],
            coordinates[indices, 2],
            c=[_record_color(records[index]) for index in indices],
            s=sizes,
            marker="o",
            alpha=0.8,
            label=source,
        )
    fall_indices = [index for index, record in enumerate(records) if record["fall"]]
    if fall_indices:
        axis.scatter(
            coordinates[fall_indices, 0],
            coordinates[fall_indices, 1],
            coordinates[fall_indices, 2],
            c="#111111",
            s=44,
            marker="x",
            label="fall",
        )
    for target in {record["target"] for record in records}:
        for source, order_key in (("cem", "iteration"), ("slerp", "fraction")):
            search_seeds = (
                {
                    record["search_seed"]
                    for record in records
                    if record["target"] == target and record["source"] == source
                }
                if source == "cem"
                else {-1}
            )
            for search_seed in search_seeds:
                indices = [
                    index
                    for index, record in enumerate(records)
                    if record["target"] == target
                    and record["source"] == source
                    and (source != "cem" or record["search_seed"] == search_seed)
                ]
                if len(indices) > 1:
                    indices.sort(key=lambda index: records[index][order_key])
                    axis.plot(
                        coordinates[indices, 0],
                        coordinates[indices, 1],
                        coordinates[indices, 2],
                        color=_record_color(records[indices[0]]),
                        linewidth=1.2,
                        alpha=0.8,
                    )
    human_push_indices = [
        index for index, record in enumerate(records) if record["source"] == "human_push_anchor"
    ]
    if len(human_push_indices) > 1:
        human_push_indices.sort(key=lambda index: records[index]["target"])
        axis.plot(
            coordinates[human_push_indices, 0],
            coordinates[human_push_indices, 1],
            coordinates[human_push_indices, 2],
            color="#2f6fed",
            linewidth=1.2,
            alpha=0.8,
        )
    axis.set_title(
        "Paper-style latent sphere\n"
        "t-SNE sphere is qualitative; quantitative distances use original latents."
    )
    axis.set_axis_off()
    axis.legend(loc="upper left", frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_score_angle(
    records: list[dict[str, Any]],
    anchors: dict[str, torch.Tensor],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    markers = {"global": "o", "geodesic": "^", "cem": "s", "slerp": "D"}
    for target, anchor in anchors.items():
        target_records = [
            record
            for record in records
            if record["target"] == target and record["source"] in markers
        ]
        if not target_records:
            continue
        latent = torch.from_numpy(np.stack([record["latent"] for record in target_records]))
        repeated_anchor = anchor.detach().cpu().reshape(1, -1).repeat(len(latent), 1)
        angles = torch.rad2deg(angular_distance(latent, repeated_anchor)).numpy()
        for source in markers:
            indices = [
                index for index, record in enumerate(target_records) if record["source"] == source
            ]
            if indices:
                axis.scatter(
                    angles[indices],
                    [target_records[index]["score"] for index in indices],
                    marker=markers[source],
                    alpha=0.65,
                    label=f"{target}: {source}",
                )
    axis.set_xlabel("Angular distance from searched anchor (degrees)")
    axis.set_ylabel("Expert target score")
    axis.set_title("Score versus original-space geodesic angle")
    axis.legend(fontsize=7, frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_support(
    support_by_target: dict[str, list[dict[str, float]]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for target, rows in support_by_target.items():
        angles = [row["angle_degrees"] for row in rows]
        axes[0].plot(angles, [row["success_rate"] for row in rows], marker="o", label=target)
        axes[1].plot(angles, [row["mean_score"] for row in rows], marker="o", label=target)
        axes[2].plot(angles, [row["fall_rate"] for row in rows], marker="o", label=target)
    axes[0].set_ylabel("Threshold success rate")
    axes[1].set_ylabel("Mean expert score")
    axes[2].set_ylabel("Fall rate")
    for axis in axes:
        axis.set_xlabel("Geodesic angle (degrees)")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7, frameon=False)
    figure.suptitle("Expert-anchor geodesic support")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _make_plots(
    records: list[dict[str, Any]],
    anchors: dict[str, torch.Tensor],
    support_by_target: dict[str, list[dict[str, float]]],
    plot_dir: Path,
    config: dict[str, Any],
    logger: logging.Logger,
) -> list[str]:
    errors = []
    operations = (
        (
            _plot_latent_tsne,
            (records, plot_dir / "latent_tsne_2d.png", config["tsne_random_state"]),
        ),
        (
            _plot_latent_sphere,
            (
                records,
                plot_dir / "latent_sphere_tsne.png",
                config["tsne_random_state"],
            ),
        ),
        (
            _plot_score_angle,
            (records, anchors, plot_dir / "score_vs_geodesic_angle.png"),
        ),
        (
            _plot_support,
            (support_by_target, plot_dir / "geodesic_support_curve.png"),
        ),
    )
    for operation, arguments in operations:
        try:
            operation(*arguments)
        except Exception as exc:
            message = f"{operation.__name__} failed: {type(exc).__name__}: {exc}"
            errors.append(message)
            logger.exception(message)
    return errors


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import imageio.v3 as iio

    iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264")


def _save_representative_videos(
    runner: H1RolloutRunner,
    selected: dict[str, tuple[torch.Tensor, ExpertTarget]],
    video_dir: Path,
    seed: int,
    logger: logging.Logger,
) -> list[str]:
    failures = []
    for name, (latent, _) in selected.items():
        try:
            rollout = runner.rollout(latent, seed=seed, capture_frames=True)
            _write_video(
                video_dir / f"{_safe_key(name)}_seed_{seed}.mp4",
                rollout.frames,
                1.0 / runner.control_dt,
            )
        except Exception as exc:
            message = f"video {name} failed: {type(exc).__name__}: {exc}"
            failures.append(message)
            logger.exception(message)
    return failures


def _summary_markdown(
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
    limitations: list[str],
) -> str:
    lines = [
        f"# {EXPERIMENT_TYPE}",
        "",
        f"- Experiment: `{metadata['experiment_name']}`",
        f"- Run type: `{metadata['run_type']}`",
        f"- Checkpoint: `{metadata['checkpoint']}`",
        f"- Result directory: `{metadata['result_directory']}`",
        "",
    ]
    if metadata["run_type"] == "smoke":
        lines.extend(
            [
                "> Smoke-only pipeline validation with a temporary checkpoint. "
                "The values below are not scientific results.",
                "",
            ]
        )
    lines.extend(
        [
            "## Dataset status",
            "",
            "| Dataset | Shape | Scoring enabled | BFM input |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in (
        "push_start_pose",
        "steer_start_pose",
        "human_push_1",
        "human_push_2",
    ):
        value = schema[name]
        lines.append(
            f"| {name} | `{value['shape']}` | {value.get('scoring_enabled', False)} | false |"
        )
    lines.extend(
        [
            "",
            "## Coverage results",
            "",
            "| Expert target | Encoded anchor | Global best | CEM best | "
            "Robust success | Angular support | Coverage type |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['target']} | false | {row['global_best_score']:.6f} | "
            f"{row['searched_anchor_score']:.6f} | "
            f"{row['robust_success_rate']:.3f} | "
            f"{row['angular_support']:.3f} | {row['coverage_type']} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in limitations)
    return "\n".join(lines) + "\n"


def _append_experiment_log(
    path: Path,
    metadata: dict[str, Any],
    config: dict[str, Any],
    schema: dict[str, Any],
    *,
    main_status: str,
    ruff_status: str,
    pytest_status: str,
    limitations: list[str],
) -> None:
    enabled = ", ".join(schema.get("enabled_targets", [])) or "none"
    unsupported = (
        "encoded expert anchors (incomplete BFM0 observations)"
        if not schema.get("encoded_anchor_available")
        else "none"
    )
    start = datetime.fromisoformat(metadata["start_time"])
    block = [
        "",
        f"## {start.date().isoformat()}",
        "",
        f"### {metadata['experiment_name']}",
        "",
        f"- Experiment type: {EXPERIMENT_TYPE}",
        f"- Run type: {metadata['run_type']}",
        f"- Start time: {metadata['start_time']}",
        f"- End time: {metadata['end_time']}",
        f"- Duration: {metadata['duration_seconds']:.3f} seconds",
        f"- Checkpoint: `{metadata['checkpoint']}`",
        f"- Git commit: `{metadata['git_commit']}`",
        "- Configuration:",
        f"  - global latents: {config['global_scan']['num_latents']}",
        f"  - CEM population: {config['cem']['population_size']}",
        f"  - CEM iterations: {config['cem']['num_iterations']}",
        f"  - horizon: {config['rollout']['horizon_seconds']} seconds",
        f"  - seeds: {config['rollout']['seeds']}",
        f"- Enabled expert targets: {enabled}",
        f"- Unsupported expert targets: {unsupported}",
        f"- Result directory: `{metadata['result_directory']}/`",
        f"- Ruff: {ruff_status}",
        f"- Pytest: {pytest_status}",
        f"- Main status: {main_status}",
        f"- Known limitations: {'; '.join(limitations)}",
    ]
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(block) + "\n")


def _append_formal_results(
    path: Path,
    metadata: dict[str, Any],
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
    latent_metrics: dict[str, Any],
    limitations: list[str],
) -> None:
    lines = [
        "",
        f"# {EXPERIMENT_TYPE}",
        "",
        f"## {metadata['experiment_name']}",
        "",
        "### Experiment metadata",
        "",
        f"- Experiment name: {metadata['experiment_name']}",
        f"- Experiment type: {EXPERIMENT_TYPE}",
        f"- Start time: {metadata['start_time']}",
        f"- End time: {metadata['end_time']}",
        f"- Duration: {metadata['duration_seconds']:.3f} seconds",
        f"- Git commit: `{metadata['git_commit']}`",
        f"- Checkpoint: `{metadata['checkpoint']}`",
        f"- Device: `{metadata['device']}`",
        f"- Result directory: `{metadata['result_directory']}`",
        "",
        "### Expert dataset status",
        "",
        "| Dataset | Format confirmed | Used as BFM input | Used as search target | Limitation |",
        "|---|---:|---:|---:|---|",
    ]
    for name in (
        "push_start_pose",
        "steer_start_pose",
        "human_push_1",
        "human_push_2",
    ):
        value = schema[name]
        lines.append(
            f"| {name} | true | false | {value.get('scoring_enabled', False)} | "
            "Incomplete BFM0 observation |"
        )
    lines.extend(
        [
            "",
            "### Configuration",
            "",
            f"- Global sphere samples: {config['global_scan']['num_latents']}",
            f"- CEM population: {config['cem']['population_size']}",
            f"- CEM iterations: {config['cem']['num_iterations']}",
            f"- Horizon: {config['rollout']['horizon_seconds']}",
            f"- Seeds: {config['rollout']['seeds']}",
            f"- Robust trials: {config['robustness']['trials']}",
            f"- Geodesic angles: {config['geodesic']['angles_degrees']}",
            f"- Samples per angle: {config['geodesic']['samples_per_angle']}",
            f"- Action gain: {config['rollout']['action_gain']}",
            "",
            "### Main results",
            "",
            "| Expert target | Encoded anchor | Global best | CEM best | "
            "Robust success | Coverage type |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['target']} | n/a | {row['global_best_score']:.6f} | "
            f"{row['searched_anchor_score']:.6f} | "
            f"{row['robust_success_rate']:.3f} | {row['coverage_type']} |"
        )
    lines.extend(["", "### Latent-space results", "", "| Metric | Result |", "|---|---:|"])
    for name, value in latent_metrics.items():
        lines.append(f"| {name} | {value} |")
    lines.extend(
        [
            "",
            "### Main findings",
            "",
            "See the per-target coverage classifications above. No encoded zero-shot "
            "conclusion is reported because the expert files do not contain complete "
            "BFM0 observations.",
            "",
            "### Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in limitations)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def _spearman_latent_behavior(
    records: list[dict[str, Any]],
    global_results: list[ScoredRollout],
) -> float:
    if not global_results:
        return math.nan
    target_name = global_results[0].target_name
    global_latents = np.stack(
        [
            record["latent"]
            for record in records
            if record["source"] == "global" and record["target"] == target_name
        ]
    )
    if len(global_latents) < 3 or len(global_results) != len(global_latents):
        return math.nan
    latent_distances = []
    behavior_distances = []
    descriptors = np.asarray(
        [list(result.rollout.descriptor.values()) for result in global_results]
    )
    for first in range(len(global_latents)):
        for second in range(first + 1, len(global_latents)):
            cosine = np.dot(global_latents[first], global_latents[second]) / (
                np.linalg.norm(global_latents[first]) * np.linalg.norm(global_latents[second])
            )
            latent_distances.append(math.acos(float(np.clip(cosine, -1.0, 1.0))))
            behavior_distances.append(
                float(np.linalg.norm(descriptors[first] - descriptors[second]))
            )
    latent_ranks = np.argsort(np.argsort(np.asarray(latent_distances)))
    behavior_ranks = np.argsort(np.argsort(np.asarray(behavior_distances)))
    return float(np.corrcoef(latent_ranks, behavior_ranks)[0, 1])


def _run_experiment(
    config: dict[str, Any],
    checkpoint: Path,
    result_dir: Path,
    metadata: dict[str, Any],
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_xml = REPO_ROOT / "husky_sim/upstream/test_scene/mjlab_scene.xml"
    dataset_root = _resolve_path(config["expert_data"]["root"])
    targets, schema = load_expert_targets(
        dataset_root,
        scene_xml,
        config["expert_data"],
    )
    _json_dump(result_dir / "dataset_schema.json", schema)
    logger.info("Confirmed expert schema; enabled targets: %s", schema["enabled_targets"])

    model, compatibility = load_bfm0_checkpoint(
        checkpoint,
        device=metadata["device"],
        run_type=metadata["run_type"],
    )
    _json_dump(result_dir / "checkpoint_compatibility.json", compatibility)
    logger.info("Checkpoint loaded strictly and model frozen")

    runner = H1RolloutRunner(model, config, device=metadata["device"])
    latent_records: list[dict[str, Any]] = []
    trajectories: dict[str, ScoredRollout] = {}
    support_by_target: dict[str, list[dict[str, float]]] = {}
    cem_history_by_target: dict[str, list[dict[str, Any]]] = {}
    anchors: dict[str, torch.Tensor] = {}
    target_rows: list[dict[str, Any]] = []
    selected_videos: dict[str, tuple[torch.Tensor, ExpertTarget]] = {}
    global_base_results: list[ScoredRollout] = []
    seed = int(config["rollout"]["seeds"][0])

    try:
        global_latents = sample_global_latents(
            model,
            int(config["global_scan"]["num_latents"]),
            seed,
        )
        global_rollouts = [
            runner.rollout(latent, seed=seed + index) for index, latent in enumerate(global_latents)
        ]
        logger.info("Completed global scan with %d latents", len(global_latents))

        global_scores_by_target: dict[str, list[ScoredRollout]] = {}
        for target in targets:
            scored = [
                score_rollout(rollout, target, config["scores"]) for rollout in global_rollouts
            ]
            global_scores_by_target[target.name] = scored
            for latent, result in zip(global_latents, scored, strict=True):
                latent_records.append(_latent_record(latent, "global", target.name, result))
            if not global_base_results:
                global_base_results = scored
        if targets:
            failure_count = 0
            for index, rollout in enumerate(global_rollouts):
                if rollout.fall and failure_count < 5:
                    selected_videos[f"failure_latent_{index:04d}"] = (
                        global_latents[index],
                        targets[0],
                    )
                    failure_count += 1

        for target_index, target in enumerate(targets):
            global_results = global_scores_by_target[target.name]
            global_best_index = int(np.argmax([result.score for result in global_results]))
            global_best = global_results[global_best_index]
            global_best_latent = global_latents[global_best_index]
            trajectories[f"{target.name}_global_best"] = global_best
            if target.kind == "static_pose":
                selected_videos[f"{target.name}_global_best"] = (
                    global_best_latent,
                    target,
                )

            def evaluate(latent: torch.Tensor, rollout_seed: int) -> ScoredRollout:
                return score_rollout(
                    runner.rollout(latent, seed=rollout_seed),
                    target,
                    config["scores"],
                )

            cem_results = []
            target_histories = []
            for search_seed_value in config["rollout"]["seeds"]:
                search_seed = int(search_seed_value)
                cem_candidate = run_cem(
                    model,
                    evaluate,
                    global_best_latent,
                    config["cem"],
                    seed=search_seed + 1000 * (target_index + 1),
                )
                cem_results.append((search_seed, cem_candidate))
                target_histories.append(
                    {
                        "search_seed": search_seed,
                        "iterations": cem_candidate.history,
                    }
                )
                for iteration, entry in enumerate(cem_candidate.history):
                    matching = max(
                        (
                            pair
                            for pair in cem_candidate.candidates
                            if np.allclose(pair[0].numpy(), entry["best_latent"])
                        ),
                        key=lambda pair: pair[1].score,
                    )
                    latent_records.append(
                        _latent_record(
                            matching[0],
                            "cem",
                            target.name,
                            matching[1],
                            iteration=iteration,
                            search_seed=search_seed,
                        )
                    )
            cem_history_by_target[target.name] = target_histories
            _, cem = max(cem_results, key=lambda item: item[1].best.score)
            anchors[target.name] = cem.best_latent
            trajectories[f"{target.name}_cem_best"] = cem.best
            selected_videos[f"{target.name}_cem_best"] = (cem.best_latent, target)
            anchor_source = (
                "human_push_anchor" if target.kind == "human_push_window" else "searched_anchor"
            )
            latent_records.append(
                _latent_record(
                    cem.best_latent,
                    anchor_source,
                    target.name,
                    cem.best,
                )
            )

            support, geodesic_records = evaluate_geodesic_support(
                model,
                cem.best_latent,
                evaluate,
                config["geodesic"],
                seed=seed + 5000 * (target_index + 1),
            )
            support_by_target[target.name] = support
            for latent, angle, result in geodesic_records:
                latent_records.append(
                    _latent_record(
                        latent,
                        "geodesic",
                        target.name,
                        result,
                        angle_degrees=angle,
                    )
                )

            robust_rate, _ = evaluate_robustness(
                runner,
                cem.best_latent,
                target,
                config["scores"],
                config["robustness"],
                seed=seed + 10000 * (target_index + 1),
            )
            global_success_rate = float(np.mean([result.success for result in global_results]))
            search_angle = float(
                angular_distance(
                    cem.best_latent.reshape(1, -1),
                    global_best_latent.reshape(1, -1),
                )[0]
            )
            small_support = next(
                (row["success_rate"] for row in support if row["angle_degrees"] <= 20.0),
                0.0,
            )
            coverage_type = classify_coverage(
                encoded_success=False,
                global_success_rate=global_success_rate,
                searched_success=cem.best.success,
                robust_success_rate=robust_rate,
                small_angle_support=small_support,
                search_angle_radians=search_angle,
                config=config["coverage"],
            )
            target_rows.append(
                {
                    "target": target.name,
                    "kind": target.kind,
                    "encoded_anchor_available": False,
                    "global_best_score": global_best.score,
                    "global_success_rate": global_success_rate,
                    "searched_anchor_score": cem.best.score,
                    "searched_success": cem.best.success,
                    "robust_success_rate": robust_rate,
                    "search_angle_degrees": math.degrees(search_angle),
                    "angular_support": small_support,
                    "coverage_type": coverage_type,
                    "global_best_metrics": global_best.metrics,
                    "searched_anchor_metrics": cem.best.metrics,
                }
            )
            logger.info(
                "Target %s: global=%.6f CEM=%.6f robust=%.3f type=%s",
                target.name,
                global_best.score,
                cem.best.score,
                robust_rate,
                coverage_type,
            )

        push_target = next(
            (target for target in targets if target.name == "push_start_pose"),
            None,
        )
        steer_target = next(
            (target for target in targets if target.name == "steer_start_pose"),
            None,
        )
        slerp_rows = []
        if push_target and steer_target:
            fractions = torch.linspace(0.0, 1.0, int(config["slerp"]["num_points"]))
            slerp_latents = spherical_lerp(
                model,
                anchors[push_target.name],
                anchors[steer_target.name],
                fractions,
            )
            representative = {0, len(fractions) // 4, len(fractions) // 2}
            representative.update({3 * len(fractions) // 4, len(fractions) - 1})
            for index, (fraction, latent) in enumerate(zip(fractions, slerp_latents, strict=True)):
                rollout = runner.rollout(latent, seed=seed + 20000 + index)
                push_score = score_rollout(rollout, push_target, config["scores"])
                steer_score = score_rollout(rollout, steer_target, config["scores"])
                slerp_rows.append(
                    {
                        "fraction": float(fraction),
                        "push_score": push_score.score,
                        "steer_score": steer_score.score,
                        "fall": rollout.fall,
                        "descriptor": rollout.descriptor,
                    }
                )
                latent_records.append(
                    _latent_record(
                        latent,
                        "slerp",
                        "push_steer_slerp",
                        push_score,
                        fraction=float(fraction),
                    )
                )
                if index in representative:
                    selected_videos[f"push_steer_slerp_t{int(100 * fraction):03d}"] = (
                        latent,
                        push_target,
                    )
            _json_dump(result_dir / "slerp_results.json", slerp_rows)

        _save_latents(result_dir / "latents.npz", latent_records)
        _save_trajectories(result_dir / "trajectories.npz", trajectories)
        _write_target_csv(result_dir / "target_results.csv", target_rows)
        plot_errors = _make_plots(
            latent_records,
            anchors,
            support_by_target,
            result_dir / "plots",
            config["visualization"],
            logger,
        )
        video_errors = []
        if config["visualization"]["save_video"]:
            video_errors = _save_representative_videos(
                runner,
                selected_videos,
                result_dir / "videos",
                seed,
                logger,
            )

        push_anchor = anchors.get("push_start_pose")
        steer_anchor = anchors.get("steer_start_pose")
        push_steer_angle = (
            float(
                torch.rad2deg(
                    angular_distance(
                        push_anchor.reshape(1, -1),
                        steer_anchor.reshape(1, -1),
                    )
                )[0]
            )
            if push_anchor is not None and steer_anchor is not None
            else math.nan
        )
        latent_metrics = {
            "Push-steer angular distance (degrees)": push_steer_angle,
            "Global stable proposal rate": float(
                np.mean([not rollout.fall for rollout in global_rollouts])
            ),
            "Latent-behavior Spearman correlation": _spearman_latent_behavior(
                latent_records,
                global_base_results,
            ),
        }
        summary = {
            "experiment_name": metadata["experiment_name"],
            "run_type": metadata["run_type"],
            "checkpoint_compatibility": compatibility,
            "encoded_anchor_available": False,
            "target_results": target_rows,
            "cem_history": cem_history_by_target,
            "geodesic_support": support_by_target,
            "latent_metrics": latent_metrics,
            "plot_errors": plot_errors,
            "video_errors": video_errors,
            "rollouts": {
                "total": runner.total_rollouts,
                "successful": runner.successful_rollouts,
                "failed": runner.failed_rollouts,
                "falls": runner.fall_count,
            },
        }
        return summary, schema
    finally:
        runner.close()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _load_config(args)
    timezone = ZoneInfo(config["experiment"]["timezone"])
    start = datetime.now(timezone)
    experiment_name = _run_name(config, args.experiment_name, start)
    output_root = _resolve_path(config["experiment"]["output_root"])
    docs_root = Path(os.environ.get("SKATE_BFM_H1_DOCS_ROOT", REPO_ROOT / "docs"))
    docs_root.mkdir(parents=True, exist_ok=True)
    result_dir = output_root / experiment_name
    result_dir.mkdir(parents=True, exist_ok=False)
    (result_dir / "plots").mkdir()
    (result_dir / "videos").mkdir()
    logger = _setup_logger(result_dir / "run.log")
    checkpoint = _resolve_path(args.checkpoint)
    git_commit, git_dirty = _git_metadata()
    device = _device(config)
    metadata: dict[str, Any] = {
        "experiment_name": experiment_name,
        "experiment_type": EXPERIMENT_TYPE,
        "run_type": config["experiment"]["run_type"],
        "start_time": start.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "checkpoint": str(checkpoint),
        "device": device,
        "result_directory": _display_path(result_dir),
    }
    _json_dump(result_dir / "metadata.json", metadata)
    (result_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Experiment %s started", experiment_name)
    logger.info("Run type=%s device=%s checkpoint=%s", metadata["run_type"], device, checkpoint)

    status = "completed"
    summary: dict[str, Any] = {}
    schema: dict[str, Any] = {}
    exit_code = 0
    started_monotonic = time.monotonic()
    limitations = [
        "Expert arrays do not provide complete BFM0 observations; encoded anchors are disabled.",
        "Static scoring uses all 30 confirmed robot bodies relative to the skateboard.",
        "Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.",
        "The current short-horizon experiment does not validate complete skateboarding.",
        "Foot contact metrics are not included in H1 coverage.",
        "t-SNE sphere plots are qualitative; quantitative distances use original latents.",
    ]
    if metadata["run_type"] == "smoke":
        limitations.insert(
            0,
            "This run uses a temporary random checkpoint and supports pipeline validation only.",
        )
    try:
        summary, schema = _run_experiment(
            config,
            checkpoint,
            result_dir,
            metadata,
            logger,
        )
    except CheckpointCompatibilityError as exc:
        status = "stopped_checkpoint_incompatible"
        exit_code = 2
        summary = {
            "experiment_name": experiment_name,
            "run_type": metadata["run_type"],
            "status": status,
            "checkpoint_compatibility": exc.report,
            "scientific_results_available": False,
        }
        _json_dump(result_dir / "checkpoint_compatibility.json", exc.report)
        schema_path = result_dir / "dataset_schema.json"
        if schema_path.exists():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        logger.error("Formal/smoke execution stopped: %s", exc)
    except Exception:
        status = "failed"
        exit_code = 1
        logger.exception("H1 execution failed")
        raise
    finally:
        end = datetime.now(timezone)
        metadata["end_time"] = end.isoformat()
        metadata["duration_seconds"] = time.monotonic() - started_monotonic
        rollouts = summary.get("rollouts", {})
        metadata.update(
            {
                "total_rollouts": rollouts.get("total", 0),
                "successful_rollouts": rollouts.get("successful", 0),
                "failed_rollouts": rollouts.get("failed", 0),
                "fall_count": rollouts.get("falls", 0),
                "status": status,
            }
        )
        _json_dump(result_dir / "metadata.json", metadata)
        summary["status"] = status
        summary["metadata"] = metadata
        _json_dump(result_dir / "summary.json", summary)
        rows = summary.get("target_results", [])
        (result_dir / "summary.md").write_text(
            _summary_markdown(metadata, rows, schema, limitations)
            if schema
            else (
                f"# {EXPERIMENT_TYPE}\n\n"
                f"Execution status: `{status}`. No scientific results were produced.\n"
            ),
            encoding="utf-8",
        )
        _append_experiment_log(
            docs_root / "exp_logs.md",
            metadata,
            config,
            schema,
            main_status=status,
            ruff_status="not run by experiment command",
            pytest_status="not run by experiment command",
            limitations=limitations,
        )
        if metadata["run_type"] == "formal" and status == "completed":
            _append_formal_results(
                docs_root / "exp_res.md",
                metadata,
                config,
                rows,
                schema,
                summary["latent_metrics"],
                limitations,
            )
        logger.info(
            "Experiment ended status=%s duration=%.3fs output=%s",
            status,
            metadata["duration_seconds"],
            metadata["result_directory"],
        )
    return exit_code


__all__ = ["main"]
