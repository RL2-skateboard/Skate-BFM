from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xml.etree.ElementTree as ET

from ..models.board import board_styles, build_board_xml


# This module deliberately keeps four concerns separate:
#   1. native MuJoCo contact construction and runtime measurements;
#   2. audited trial loading;
#   3. differentiable offline experiments;
#   4. constrained contact-model experiments.
# The last three are framework boundaries at this point. ``pass`` is intentional:
# no missing data or model is silently replaced by zeros or a fake PASS result.


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
    method: str = "cem"
    parameter_names: tuple[str, ...] = ("mu_static", "mu_dynamic")
    parameter_lower: tuple[float, ...] = (0.01, 0.01)
    parameter_upper: tuple[float, ...] = (2.0, 2.0)
    seed: int = 0


@dataclass
class ContactModel:
    """Interface for an analytic/differentiable/constrained contact model.

    Implementations must enforce: no contact -> no force, non-negative normal
    force, friction bound, and non-positive dissipative power. They must not
    add a second friction force on top of an already active engine contact pair.
    """

    name: str
    config: dict[str, Any] = field(default_factory=dict)

    def predict(self, features: Any) -> Any:
        raise NotImplementedError(f"contact model {self.name!r} has no predictor")

    def rollout(self, initial_state: Any, controls: Any, dt: float) -> Any:
        raise NotImplementedError(f"contact model {self.name!r} has no rollout")


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
    """Load and audit contact trials from ``data/contact``.

    Required behavior for the future implementation:
    - read a manifest with URL/DOI, version, license and file hashes;
    - reject missing units, frames, timestamps or force semantics;
    - split by complete trial/session, never by individual time samples;
    - never fill an unavailable measurement with zero.
    """
    del root, source_id
    pass


def audit_dataset(dataset: ContactDataset) -> dict[str, Any]:
    """Validate field meaning, units, trial splits and leakage constraints."""
    del dataset
    pass


def synthetic_trials(*, seed: int, count: int, held_out_count: int) -> ContactDataset:
    """Generate separated synthetic trials for method/gradient tests only.

    Synthetic results must be labeled as synthetic and cannot be reported as
    sim-to-real improvement. Train and held-out trials must use different
    excitation/noise/parameter combinations.
    """
    del seed, count, held_out_count
    pass


def predict_contact(model: ContactModel, features: Any) -> Any:
    """Run a force prediction with physical-constraint validation."""
    del model, features
    pass


def rollout_contact(model: ContactModel, initial_state: Any, controls: Any, dt: float) -> Any:
    """Run a multi-step trajectory rollout for ``trajectory_fit`` experiments."""
    del model, initial_state, controls, dt
    pass


def fit_contact(dataset: ContactDataset, config: ContactFitConfig) -> dict[str, Any]:
    """Fit bounded contact parameters/model; require an audited split first.

    ``force_fit`` and ``trajectory_fit`` are separate tasks. A force loss is
    never presented as evidence of a free dynamics rollout.
    """
    del dataset, config
    pass


def evaluate_contact(model: ContactModel, dataset: ContactDataset, *, split: str) -> dict[str, Any]:
    """Evaluate on complete held-out trials and report physical diagnostics."""
    del model, dataset, split
    pass


def finite_difference_check(model: ContactModel, state: Any, control: Any, dt: float) -> dict[str, Any]:
    """Compare autodiff gradients with multiple finite-difference step sizes."""
    del model, state, control, dt
    pass
