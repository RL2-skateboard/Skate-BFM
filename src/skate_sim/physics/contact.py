from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xml.etree.ElementTree as ET

from ..models.board import board_styles, build_board_xml


# Native contact and offline identification share this entry point. Torch is
# imported only when an offline model is constructed, never during native play.


Origin = Literal["synthetic", "public_real", "target_real"]
FitTask = Literal["force_fit", "trajectory_fit"]


@dataclass
class ContactTrial:
    """One complete, leakage-safe contact trial.

    Arrays are intentionally left as ``Any`` so this module does not import
    torch/JAX during normal MuJoCo startup. A later adapter must document units,
    frame, timestamps and whether each field is measured, commanded or estimated.
    """

    trial_id: str
    origin: Origin
    material_pair: str
    units: dict[str, str]
    frame: str
    time: Any = None
    foot_pose: Any = None
    foot_velocity: Any = None
    deck_pose: Any = None
    deck_velocity: Any = None
    normal_force: Any = None
    tangent_force: Any = None
    external_input: Any = None
    contact_state: Any = None
    gap: Any = None
    taxel_force: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContactDataset:
    """Audited collection of complete trials and their split assignment."""

    trials: list[ContactTrial]
    splits: dict[str, list[str]]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ContactFitConfig:
    """Configuration shared by synthetic, public-real and target-real fits."""

    task: FitTask = "force_fit"
    method: str = "adam"
    model: str = "analytic"
    epochs: int = 150
    learning_rate: float = 0.03
    seed: int = 0
    init: str | None = None


CONTACT_SCHEMA = "skate-sim.contact.v1"
MODEL_SCHEMA = "skate-sim.contact-model.v1"
PARAMETER_BOUNDS = {"stiffness": (2000., 30000.), "damping": (5., 150.), "mu": (.05, 1.5)}
DEFAULT_PARAMETERS = {"stiffness": 10000., "damping": 45., "mu": .5}
CONTACT_UNITS = {"time": "s", "gap": "m", "foot_pose": "m", "foot_velocity": "m/s",
                 "taxel_force": "N", "tangent_force": "N", "external_input": "N",
                 "taxel_positions": "m", "mass": "kg"}


@dataclass
class ContactModel:
    """Offline flat-sole contact, with optional bounded parameter residual MLP.

    This is NOT the MuJoCo/Warp solver. The differentiable rollout has three
    translational DOFs, a fixed horizontal deck, and no foot rotation. Its
    compliance parameters cannot be copied into MuJoCo solref/solimp.
    """

    name: str
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import torch
        if self.name not in ("analytic", "contact_net"):
            raise ValueError("contact model must be analytic or contact_net")
        self.module = torch.nn.Module()
        self.module.register_parameter("raw", torch.nn.Parameter(torch.zeros(3, dtype=torch.float64)))
        if self.name == "contact_net":
            net = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3)).double()
            torch.nn.init.zeros_(net[-1].weight)
            torch.nn.init.zeros_(net[-1].bias)
            self.module.add_module("residual", net)
            # Fit the analytic baseline first; the residual must not disguise
            # a freely drifting global parameter estimate.
            self.module.raw.requires_grad_(False)
        self.set_parameters(self.config.get("parameters", DEFAULT_PARAMETERS))

    def set_parameters(self, values: dict[str, float]) -> None:
        import torch
        if set(values) != set(PARAMETER_BOUNDS):
            raise ValueError("contact parameters require stiffness, damping and mu")
        ratios = []
        for name, (low, high) in PARAMETER_BOUNDS.items():
            value = float(values[name])
            if not np.isfinite(value) or not low < value < high:
                raise ValueError(f"{name} must be inside ({low}, {high})")
            ratios.append((value - low) / (high - low))
        with torch.no_grad():
            self.module.raw.copy_(torch.logit(torch.tensor(ratios, dtype=torch.float64)))

    def parameters(self):
        import torch
        low, high = np.asarray(list(PARAMETER_BOUNDS.values())).T
        return torch.as_tensor(low) + torch.sigmoid(self.module.raw) * torch.as_tensor(high - low)

    def parameter_values(self) -> dict[str, float]:
        return dict(zip(PARAMETER_BOUNDS, self.parameters().detach().tolist()))

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        import torch
        gap = torch.as_tensor(features["gap"], dtype=torch.float64)
        velocity = torch.as_tensor(features["velocity"], dtype=torch.float64)
        xy = torch.as_tensor(features["taxel_positions"], dtype=torch.float64)
        k, c, mu = self.parameters().unbind()
        if self.name == "contact_net":
            v = velocity[..., None, :].expand(*gap.shape, 3)
            points = xy.expand(*gap.shape, 2)
            inputs = torch.cat((gap[..., None] / .005, v / .2, points / .1), dim=-1)
            # Bounded inputs and outputs also bound extrapolation. The MLP
            # never reads measured force labels, mass or a current solver force.
            residual = .2 * torch.tanh(self.module.residual(inputs.clamp(-5, 5)))
            k = (k * (1 + residual[..., 0])).clamp(*PARAMETER_BOUNDS["stiffness"])
            c = (c * (1 + residual[..., 1])).clamp(*PARAMETER_BOUNDS["damping"])
            mu = (mu * (1 + residual[..., 2])).clamp(*PARAMETER_BOUNDS["mu"])
        penetration = (-gap).clamp_min(0)
        active = gap < 0
        elastic = k * penetration / gap.shape[-1]
        normal = ((k * penetration - c * velocity[..., 2:3]) / gap.shape[-1]).clamp_min(0) * active
        vt = velocity[..., :2]
        speed = torch.sqrt((vt * vt).sum(-1, keepdim=True) + .01 ** 2)
        tangent = -(mu * normal).sum(-1, keepdim=True) * vt / speed
        total = normal.sum(-1)
        cop = (normal[..., None] * xy).sum(-2) / total.clamp_min(1e-12)[..., None]
        return {"taxel_force": normal, "normal_force": total, "tangent_force": tangent,
                "tangent_drag": (mu * normal).sum(-1) / speed.squeeze(-1),
                "cop": cop, "cop_valid": total > 1e-9,
                "area_ratio": (normal > .1).double().mean(-1),
                "tangent_power": (tangent * vt).sum(-1),
                "damping_power": ((normal - elastic) * velocity[..., 2:3]).sum(-1)}

    def rollout(self, initial_state: Any, controls: Any, dt: float) -> Any:
        import torch
        if not np.isfinite(dt) or not 0 < dt <= .002:
            raise ValueError("offline rollout requires 0 < dt <= 0.002 seconds")
        mass = float(self.config["mass"])
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("mass must be positive")
        state = torch.as_tensor(initial_state, dtype=torch.float64)
        controls = torch.as_tensor(controls, dtype=torch.float64)
        xy = torch.as_tensor(self.config["taxel_positions"], dtype=torch.float64)
        states = [state]
        for force in controls:
            predicted = self.predict({"gap": state[2].expand(len(xy)), "velocity": state[3:],
                                      "taxel_positions": xy})
            contact = torch.cat((predicted["tangent_force"], predicted["normal_force"].reshape(1)))
            acceleration = (contact + force) / mass + state.new_tensor([0, 0, -9.81])
            # Linearized implicit friction avoids explicit-Euler sign reversal
            # when the foot slows down inside the regularized friction region.
            tangent_velocity = ((state[3:5] + dt * force[:2] / mass)
                                / (1 + dt * predicted["tangent_drag"] / mass))
            velocity = torch.cat((tangent_velocity, (state[5] + dt * acceleration[2]).reshape(1)))
            state = torch.cat((state[:3] + dt * velocity, velocity))
            states.append(state)
        return torch.stack(states)


def contact_xml(*, freeboard: bool = False) -> str:
    style = board_styles("standard")["standard"]
    root = ET.fromstring(build_board_xml(style, gallery=False))
    root.set("model", "skate-sim-contact-standard-board")
    world = root.find("worldbody")
    deck = world.find("body[@name='deck']")
    if not freeboard:
        freejoint = deck.find("joint[@name='deck_free']")
        if freejoint is not None:
            deck.remove(freejoint)
    foot = ET.SubElement(world, "body", name="foot", pos="0 0 .17")
    ET.SubElement(foot, "freejoint", name="foot_free")
    ET.SubElement(foot, "geom", name="foot_contact", type="capsule",
                  fromto="-.07 0 0 .07 0 0", size=".045", mass=".8",
                  friction=".8 .02 .002", rgba=".85 .35 .12 1")
    ET.SubElement(foot, "site", name="foot_center", size=".008", rgba="1 1 0 1")
    return ET.tostring(root, encoding="unicode")


def contact_metrics(model, data, foot_id: int, deck_id: int) -> dict:
    slip = data.cvel[foot_id, 3:] - data.cvel[deck_id, 3:]
    tangent_speed = float(np.linalg.norm(slip[:2]))
    normal_force = 0.0
    tangential_force = 0.0
    contacts = 0
    for i in range(data.ncon):
        c = data.contact[i]
        bodies = (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
        if foot_id not in bodies or deck_id not in bodies:
            continue
        force = np.zeros(6)
        import mujoco
        mujoco.mj_contactForce(model, data, i, force)
        normal_force += max(0.0, float(force[0]))
        tangential_force += float(np.linalg.norm(force[1:3]))
        contacts += 1
    return {"contacts": contacts, "normal_force_N": normal_force,
            "tangential_force_N": tangential_force,
            "relative_slip_speed_m_s": tangent_speed,
            "friction_power_W": max(0.0, tangential_force * tangent_speed)}


def load_trials(root: str | Path, *, source_id: str | None = None) -> ContactDataset:
    """Read a self-contained, SI-normalized JSON file; never infer missing data."""
    path = Path(root)
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema") != CONTACT_SCHEMA:
        raise ValueError(f"expected dataset schema {CONTACT_SCHEMA}")
    if source_id is not None and data["manifest"].get("source_id") != source_id:
        raise ValueError("source_id does not match manifest")
    dataset = ContactDataset([ContactTrial(**row) for row in data["trials"]],
                             data["splits"], data["manifest"])
    audit_dataset(dataset, dataset.manifest.get("task", "force_fit"))
    dataset.manifest["input_sha256"] = hashlib.sha256(raw).hexdigest()
    return dataset


def audit_dataset(dataset: ContactDataset, task: FitTask = "force_fit") -> dict[str, Any]:
    """Strict v1 contract: one material pair, flat fixed deck, complete sessions."""
    if task not in ("force_fit", "trajectory_fit"):
        raise ValueError("unknown contact fitting task")
    manifest = dataset.manifest
    if manifest.get("task") != task:
        raise ValueError(f"manifest task must match requested task {task!r}")
    for key in ("source_id", "source", "version", "license", "force_semantics"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"manifest requires {key}")
    if manifest["force_semantics"] != "force_on_foot_deck_frame_z_up":
        raise ValueError("force semantics must be force_on_foot_deck_frame_z_up")
    xy = np.asarray(manifest.get("taxel_positions"), dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError("taxel_positions must be finite [M,2] sole coordinates")
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError("taxel positions must be unique")
    ids = [trial.trial_id for trial in dataset.trials]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("trial IDs must be unique non-empty strings")
    if set(dataset.splits) != {"train", "validation", "test"}:
        raise ValueError("require train, validation and untouched test splits")
    assigned = [i for split in dataset.splits.values() for i in split]
    if any(not split for split in dataset.splits.values()) or sorted(assigned) != sorted(ids):
        raise ValueError("splits must partition every complete trial exactly once")
    sessions, fingerprints = {}, set()
    split_for = {i: split for split, members in dataset.splits.items() for i in members}
    for trial in dataset.trials:
        if trial.origin not in ("synthetic", "public_real", "target_real"):
            raise ValueError("trial origin must explicitly identify synthetic or real data")
        if trial.frame != "fixed_deck_z_up" or trial.units != CONTACT_UNITS:
            raise ValueError(f"{trial.trial_id}: require documented fixed_deck_z_up frame and exact SI units")
        session = trial.metadata.get("session_id")
        if not isinstance(session, str) or not session:
            raise ValueError("each trial requires session_id")
        if session in sessions and sessions[session] != split_for[trial.trial_id]:
            raise ValueError("session leakage across splits")
        sessions[session] = split_for[trial.trial_id]
        if trial.origin != "synthetic":
            for key in ("measurement_error", "synchronization", "preprocessing", "field_roles"):
                if not trial.metadata.get(key):
                    raise ValueError(f"real trial requires metadata.{key}")
        time = np.asarray(trial.time, dtype=float)
        if time.ndim != 1 or len(time) < 3 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError("timestamps must be finite and strictly increasing")
        shapes = {"gap": (len(time), len(xy)), "foot_velocity": (len(time), 3)}
        for name, shape in (("taxel_force", (len(time), len(xy))), ("tangent_force", (len(time), 2))):
            if task == "force_fit" or getattr(trial, name) is not None:
                shapes[name] = shape
        if task == "trajectory_fit":
            shapes.update(foot_pose=(len(time), 3), external_input=(len(time) - 1, 3))
            mass = float(trial.metadata.get("mass", 0))
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("trajectory fitting requires a measured positive mass")
            if not np.allclose(np.diff(time), time[1] - time[0], rtol=1e-6, atol=1e-12) or time[1] - time[0] > .002:
                raise ValueError("trajectory timestamps require uniform dt <= .002 s; resampling must be audited")
        for name, shape in shapes.items():
            values = np.asarray(getattr(trial, name), dtype=float)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"{trial.trial_id}: {name} must be finite with shape {shape}; no missing labels")
        if trial.taxel_force is not None and np.any(np.asarray(trial.taxel_force) < 0):
            raise ValueError("normal taxel forces must be non-negative")
        if task == "trajectory_fit" and not np.allclose(np.asarray(trial.gap), np.asarray(trial.foot_pose)[:, 2:3], atol=1e-9):
            raise ValueError("v1 rollout supports a flat non-rotating sole on a fixed deck only")
        fingerprint = hashlib.sha256(json.dumps(
            [trial.time, trial.gap, trial.foot_velocity, trial.foot_pose, trial.external_input],
            sort_keys=True).encode()).hexdigest()
        if fingerprint in fingerprints:
            raise ValueError("duplicate trial content, including renamed trials")
        fingerprints.add(fingerprint)
    if len({t.material_pair for t in dataset.trials}) != 1 or not dataset.trials[0].material_pair:
        raise ValueError("v1 fits one explicitly named material pair per dataset")
    if len({t.origin for t in dataset.trials}) != 1:
        raise ValueError("do not mix synthetic and real origins in one fit")
    return {"status": "PASS", "origin": dataset.trials[0].origin,
            "trials": len(ids), "splits": {k: len(v) for k, v in dataset.splits.items()}}


def synthetic_trials(*, seed: int, count: int = 4, held_out_count: int = 2,
                     task: FitTask = "force_fit") -> ContactDataset:
    """Same-model method validation, with independent excitations and force noise.

    The material parameters are shared across trials to make recovery meaningful.
    This does not test structural mismatch or real sim2real generalization.
    """
    import torch
    if count < 1 or held_out_count < 1 or task not in ("force_fit", "trajectory_fit"):
        raise ValueError("positive split sizes and a supported task are required")
    rng = np.random.default_rng(seed)
    xy = [[float(x), float(y)] for x in np.linspace(-.09, .09, 4) for y in (-.035, 0., .035)]
    truth = {"stiffness": 16000., "damping": 85., "mu": .65}
    model = ContactModel("analytic", {"parameters": truth, "mass": .8, "taxel_positions": xy})
    trials, splits = [], {"train": [], "validation": [], "test": []}
    for split, number in (("train", count), ("validation", held_out_count), ("test", held_out_count)):
        for index in range(number):
            time = np.arange(96) * .001
            if task == "trajectory_fit":
                force = np.zeros((len(time) - 1, 3))
                force[:, 0] = rng.uniform(-2, 2)
                initial = [0, 0, rng.uniform(.001, .004), rng.uniform(.1, .4), rng.uniform(-.1, .1), -.15]
                with torch.no_grad():
                    state = model.rollout(initial, force, .001).numpy()
                pose, velocity = state[:, :3], state[:, 3:]
                gap = np.repeat(pose[:, 2:3], len(xy), axis=1)
                external = force.tolist()
            else:
                frequency, phase = rng.uniform(8, 16), rng.uniform(0, 2 * np.pi)
                z = -.001 + .002 * np.sin(frequency * time + phase)
                gap = z[:, None] + np.asarray(xy)[:, 0] * rng.uniform(-.01, .01)
                velocity = np.column_stack((rng.uniform(.02, .4) * np.ones_like(time),
                                            .1 * np.sin(25 * time + phase),
                                            .002 * frequency * np.cos(frequency * time + phase)))
                pose, external = None, None
            with torch.no_grad():
                pred = model.predict({"gap": gap, "velocity": velocity, "taxel_positions": xy})
            # Independent noise per complete trial; absent contacts stay absent.
            normal = pred["taxel_force"].numpy()
            normal = np.maximum(0, normal + rng.normal(0, .002, normal.shape) * (normal > 0))
            tangent = pred["tangent_force"].numpy() + rng.normal(0, .002, (len(time), 2))
            identifier = f"{split}-{index:03d}"
            trials.append(ContactTrial(identifier, "synthetic", "synthetic_sole_deck", dict(CONTACT_UNITS),
                                       "fixed_deck_z_up", time=time.tolist(),
                                       foot_pose=None if pose is None else pose.tolist(),
                                       foot_velocity=velocity.tolist(), gap=gap.tolist(),
                                       taxel_force=normal.tolist(), tangent_force=tangent.tolist(),
                                       external_input=external,
                                       metadata={"session_id": identifier, "mass": .8}))
            splits[split].append(identifier)
    return ContactDataset(trials, splits, {
        "source_id": "synthetic-contact", "source": "skate_sim.physics.contact.synthetic_trials",
        "version": CONTACT_SCHEMA, "license": "generated locally; no third-party data",
        "force_semantics": "force_on_foot_deck_frame_z_up", "taxel_positions": xy,
        "truth": truth, "seed": seed, "task": task,
        "disclosure": "same reduced model generator; no real data or engine-gradient claim"})


def _dataset_signature(dataset: ContactDataset) -> str:
    data = asdict(dataset)
    data["manifest"].pop("input_sha256", None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def predict_contact(model: ContactModel, features: Any) -> Any:
    """Common forward operation for fitting, evaluation and future adapters."""
    return model.predict(features)


def rollout_contact(model: ContactModel, initial_state: Any, controls: Any, dt: float) -> Any:
    return model.rollout(initial_state, controls, dt)


def _prediction(model: ContactModel, trial: ContactTrial, xy: Any, task: str):
    import torch
    state = None
    if task == "trajectory_fit":
        model.config.update(mass=trial.metadata["mass"], taxel_positions=xy)
        state = rollout_contact(model, [*trial.foot_pose[0], *trial.foot_velocity[0]],
                                trial.external_input, trial.time[1] - trial.time[0])
        gap, velocity = state[:, 2:3].expand(-1, len(xy)), state[:, 3:]
    else:
        gap, velocity = trial.gap, trial.foot_velocity
    pred = predict_contact(model, {"gap": gap, "velocity": velocity, "taxel_positions": xy})
    if task == "force_fit":
        loss = ((pred["taxel_force"] - torch.as_tensor(trial.taxel_force)) / .5).square().mean()
        loss = loss + ((pred["tangent_force"] - torch.as_tensor(trial.tangent_force)) / 5.).square().mean()
    else:
        # Open loop: after t=0 no measured state is injected into the rollout.
        target = torch.cat((torch.as_tensor(trial.foot_pose), torch.as_tensor(trial.foot_velocity)), -1)
        scale = state.new_tensor([.005] * 3 + [.1] * 3)
        loss = ((state[1:] - target[1:]) / scale).square().mean()
    return loss, pred, state


def fit_contact(dataset: ContactDataset, config: ContactFitConfig) -> dict[str, Any]:
    """Adam on train only; select on validation; evaluate test once at the end."""
    import torch
    audit = audit_dataset(dataset, config.task)
    if config.method != "adam" or config.epochs < 1 or not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("require adam, positive epochs and a finite positive learning rate")
    if config.seed < 0:
        raise ValueError("seed must be non-negative")
    if config.model == "contact_net" and not config.init:
        raise ValueError("contact_net requires --init from an analytic fit; fit the baseline first")
    torch.manual_seed(config.seed)
    model = ContactModel(config.model, {"task": config.task})
    if config.init:
        initial = load_contact_model(config.init)
        if initial.name != "analytic":
            raise ValueError("--init must be an analytic baseline")
        if initial.config.get("material_pair") != dataset.trials[0].material_pair:
            raise ValueError("initial model material pair does not match dataset")
        if initial.config.get("dataset_signature") != _dataset_signature(dataset):
            raise ValueError("initial baseline must use the same audited dataset and splits")
        model.set_parameters(initial.parameter_values())
    model.config.update(task=config.task, material_pair=dataset.trials[0].material_pair,
                        dataset_signature=_dataset_signature(dataset))
    xy = dataset.manifest["taxel_positions"]
    baseline = {split: evaluate_contact(model, dataset, split=split) for split in ("train", "validation", "test")}
    train = [trial for trial in dataset.trials if trial.trial_id in dataset.splits["train"]]
    optimizer = torch.optim.Adam([p for p in model.module.parameters() if p.requires_grad], lr=config.learning_rate)
    best = copy.deepcopy(model.module.state_dict())
    best_loss, best_epoch, history = baseline["validation"]["loss"], 0, []
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        losses = [_prediction(model, trial, xy, config.task)[0] for trial in train]
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.module.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        validation = evaluate_contact(model, dataset, split="validation")["loss"]
        history.append({"epoch": epoch, "train_loss_before_step": float(loss.detach()), "validation_loss": validation})
        if validation < best_loss:
            best_loss, best_epoch, best = validation, epoch, copy.deepcopy(model.module.state_dict())
        if epoch == 1 or epoch % 25 == 0 or epoch == config.epochs:
            print(f"epoch {epoch}/{config.epochs}: train={float(loss.detach()):.6g} validation={validation:.6g}", flush=True)
    model.module.load_state_dict(best)
    metrics = {split: evaluate_contact(model, dataset, split=split) for split in ("train", "validation", "test")}
    first = train[0]
    model.config.update(mass=float(first.metadata.get("mass", .8)), taxel_positions=xy)
    gradient = finite_difference_check(model, [0, 0, -.0008, .15, .03, -.02], [[0, 0, 0]] * 12, .001)
    test_baseline_loss = baseline["test"]["loss"]
    checks = {"finite_metrics": all(np.isfinite(row["loss"]) for row in metrics.values()),
              "physical_constraints": all(row["constraints_ok"] for row in metrics.values()),
              "gradient_check": gradient["status"] == "PASS",
              "test_improves_baseline": metrics["test"]["loss"] < test_baseline_loss}
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "scope": "offline reduced contact model; not deployed to native backends",
              "origin": audit["origin"], "config": asdict(config), "audit": audit,
              "parameters": model.parameter_values(), "baseline": baseline, "metrics": metrics,
              "best_epoch": best_epoch, "history": history, "gradient_check": gradient,
              "split_ids": dataset.splits, "bounds_source": "engineering assumptions"}
    if audit["origin"] == "synthetic":
        truth = dataset.manifest.get("truth")
        if truth:
            report["relative_parameter_errors"] = {k: abs(report["parameters"][k] / truth[k] - 1) for k in truth}
        report["disclosure"] = "synthetic method validation only; no real sim2real improvement"
    if model.name == "contact_net":
        report["network_comparison"] = {
            "analytic_baseline_test_loss": test_baseline_loss,
            "network_test_loss": metrics["test"]["loss"],
            "relative_change": metrics["test"]["loss"] / max(test_baseline_loss, 1e-12) - 1,
            "interpretation": "no improvement is a valid experimental result but does not pass the improvement gate",
        }
    return {"model": model, "report": report}


def evaluate_contact(model: ContactModel, dataset: ContactDataset, *, split: str) -> dict[str, Any]:
    import torch
    if split not in dataset.splits:
        raise ValueError(f"unknown split {split}")
    rows = []
    with torch.no_grad():
        for trial in dataset.trials:
            if trial.trial_id not in dataset.splits[split]:
                continue
            loss, pred, state = _prediction(model, trial, dataset.manifest["taxel_positions"], model.config.get("task", "force_fit"))
            normal = pred["taxel_force"]
            checks = (torch.isfinite(normal).all() and torch.isfinite(pred["tangent_force"]).all()
                      and (normal >= 0).all() and (pred["tangent_power"] <= 1e-9).all()
                      and (pred["damping_power"] <= 1e-9).all()
                      and (torch.linalg.vector_norm(pred["tangent_force"], dim=-1) <= 1.5 * pred["normal_force"] + 1e-9).all())
            gap = torch.as_tensor(trial.gap) if state is None else state[:, 2:3].expand_as(normal)
            checks = checks and (normal[gap >= 0] == 0).all()
            rows.append({"trial_id": trial.trial_id, "loss": float(loss), "constraints_ok": bool(checks),
                         "taxel_rmse_N": None if trial.taxel_force is None else float((normal - torch.as_tensor(trial.taxel_force)).square().mean().sqrt()),
                         "tangent_rmse_N": None if trial.tangent_force is None else float((pred["tangent_force"] - torch.as_tensor(trial.tangent_force)).square().mean().sqrt()),
                         "position_rmse_m": None if state is None else float((state[:, :3] - torch.as_tensor(trial.foot_pose)).square().mean().sqrt())})
    if not rows:
        raise ValueError("cannot evaluate an empty split")
    return {"loss": float(np.mean([r["loss"] for r in rows])),
            "constraints_ok": all(r["constraints_ok"] for r in rows), "trials": rows}


def finite_difference_check(model: ContactModel, state: Any, control: Any, dt: float) -> dict[str, Any]:
    """Directional derivative through the full multi-step rollout, float64.

    Contact switching is piecewise differentiable, not globally smooth. The
    smooth probe here does not certify gradients at impact or zero penetration.
    """
    import torch
    parameters = [p for p in model.module.parameters() if p.requires_grad]
    directions = [torch.cos(torch.arange(p.numel(), dtype=torch.float64)).reshape_as(p) for p in parameters]
    norm = sum(d.square().sum() for d in directions).sqrt()
    directions = [d / norm for d in directions]
    def objective():
        trajectory = model.rollout(state, control, dt)
        return (trajectory[:, 2] / .005).square().mean() + trajectory[:, 3:].square().mean()
    original = [p.detach().clone() for p in parameters]
    gradients = torch.autograd.grad(objective(), parameters)
    autodiff = float(sum((g * d).sum() for g, d in zip(gradients, directions)))
    results = []
    try:
        for eps in (1e-3, 1e-4, 1e-5):
            values = []
            for sign in (-1, 1):
                with torch.no_grad():
                    for p, value, direction in zip(parameters, original, directions):
                        p.copy_(value + sign * eps * direction)
                    values.append(float(objective()))
            finite = (values[1] - values[0]) / (2 * eps)
            error = abs(finite - autodiff)
            results.append({"epsilon": eps, "finite_difference": finite,
                            "absolute_error": error, "pass": error <= 1e-6 + 1e-3 * abs(autodiff)})
    finally:
        with torch.no_grad():
            for p, value in zip(parameters, original):
                p.copy_(value)
    return {"status": "PASS" if all(r["pass"] for r in results) else "FAIL",
            "autodiff": autodiff, "probes": results,
            "contact_switches": "not differentiable at gap=0 or normal-force clamp; use one-sided diagnostics"}


def save_contact_run(output: str | Path, dataset: ContactDataset, result: dict[str, Any]) -> Path:
    """Separate offline artifacts from native Stage 09 params.yaml on purpose."""
    import torch
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    for name in ("dataset.json", "report.json", "model.pt", "contact_parameters.json"):
        if (path / name).exists():
            raise ValueError(f"output already contains {name}; choose a new directory")
    model, report = result["model"], result["report"]
    data = {"schema": CONTACT_SCHEMA, **asdict(dataset)}
    raw = json.dumps(data, indent=2, allow_nan=False) + "\n"
    report["dataset_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    (path / "dataset.json").write_text(raw, encoding="utf-8")
    payload = {"schema": MODEL_SCHEMA, "model": model.name, "config": model.config,
               "state_dict": model.module.state_dict()}
    torch.save(payload, path / "model.pt")
    params = {"schema": MODEL_SCHEMA, "parameters": model.parameter_values(),
              "units": {"stiffness": "N/m", "damping": "N*s/m", "mu": "1"},
              "bounds": PARAMETER_BOUNDS, "origin": report["origin"], "native_backend_compatible": False}
    (path / "contact_parameters.json").write_text(json.dumps(params, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (path / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def load_contact_model(path: str | Path) -> ContactModel:
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported contact model schema")
    model = ContactModel(payload["model"], payload["config"])
    model.module.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.module.parameters()):
        raise ValueError("non-finite contact model weights")
    return model


def contact_figure(run: str | Path, trial_id: str | None = None):
    """Build a measured/default-or-fitted-baseline/candidate comparison for play."""
    import matplotlib.pyplot as plt
    import torch
    root = Path(run)
    dataset = load_trials(root / "dataset.json")
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    model = load_contact_model(root / "model.pt")
    if (dataset.manifest["input_sha256"] != report["dataset_sha256"]
            or model.config.get("dataset_signature") != _dataset_signature(dataset)):
        raise ValueError("run dataset was modified or does not match the trained model")
    identifier = trial_id or dataset.splits["test"][0]
    trial = next((t for t in dataset.trials if t.trial_id == identifier), None)
    if trial is None:
        raise ValueError(f"unknown trial {identifier}; available: {[t.trial_id for t in dataset.trials]}")
    baseline = ContactModel("analytic", {"task": model.config["task"]})
    if model.name == "contact_net":
        baseline.set_parameters(model.parameter_values())
    xy = dataset.manifest["taxel_positions"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.canvas.manager.set_window_title("skate-sim | offline contact comparison")
    for candidate, label in ((baseline, "baseline"), (model, "trained")):
        with torch.no_grad():
            _, pred, state = _prediction(candidate, trial, xy, model.config["task"])
        axes[0, 0].plot(trial.time, pred["normal_force"].numpy(), label=label)
        axes[0, 1].plot(trial.time, pred["tangent_force"][:, 0].numpy(), label=label)
        if state is not None:
            axes[1, 0].plot(trial.time, state[:, 2].numpy(), label=label)
        else:
            axes[1, 0].plot(trial.time, np.where(pred["cop_valid"], pred["cop"][:, 0], np.nan), label=label)
    if trial.taxel_force is not None:
        normal = np.asarray(trial.taxel_force)
        axes[0, 0].plot(trial.time, normal.sum(-1), "k--", label="target")
        if model.config["task"] == "force_fit":
            valid = normal.sum(-1) > 1e-9
            cop = (normal * np.asarray(xy)[:, 0]).sum(-1) / np.maximum(normal.sum(-1), 1e-12)
            axes[1, 0].plot(trial.time, np.where(valid, cop, np.nan), "k--", label="target")
    if trial.tangent_force is not None:
        axes[0, 1].plot(trial.time, np.asarray(trial.tangent_force)[:, 0], "k--", label="target")
    if model.config["task"] == "trajectory_fit":
        axes[1, 0].plot(trial.time, np.asarray(trial.foot_pose)[:, 2], "k--", label="target")
    for axis, title, unit in ((axes[0, 0], "Normal force", "N"), (axes[0, 1], "Tangential force X", "N"),
                             (axes[1, 0], "Sole height" if model.config["task"] == "trajectory_fit" else "Center of pressure X", "m")):
        axis.set(title=title, xlabel="time [s]", ylabel=unit)
        axis.legend()
        axis.grid(alpha=.25)
    epochs = [r["epoch"] for r in report["history"]]
    axes[1, 1].semilogy(epochs, [max(r["train_loss_before_step"], 1e-16) for r in report["history"]], label="train (pre-step)")
    axes[1, 1].semilogy(epochs, [max(r["validation_loss"], 1e-16) for r in report["history"]], label="validation")
    axes[1, 1].set(title=f"Test loss: {report['metrics']['test']['loss']:.4g}", xlabel="epoch", ylabel="normalized loss")
    axes[1, 1].legend()
    fig.suptitle(f"{trial.origin} | {identifier} | {model.name} | {report['status']}\n"
                 "Offline 3-DOF contact model; not native engine physics or real sim2real validation")
    fig.tight_layout()
    return fig
