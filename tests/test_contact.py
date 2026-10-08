"""Physical, leakage and gradient contracts for offline identification."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from skate_sim.physics.contact import (
    ContactModel, audit_dataset, finite_difference_check, load_contact_model,
    synthetic_trials, CONTACT_SCHEMA, load_trials,
)


class ContactTests(unittest.TestCase):
    def test_constraints_and_no_contact(self):
        for kind in ("analytic", "contact_net"):
            model = ContactModel(kind)
            gap = np.array([[.001, .003], [-.002, -.001], [-.002, -.001]])
            velocity = np.array([[.3, -.1, -.1], [.3, -.2, -.1], [.3, -.2, .1]])
            pred = model.predict({"gap": gap, "velocity": velocity, "taxel_positions": [[-.05, 0], [.05, 0]]})
            self.assertTrue(torch.equal(pred["taxel_force"][0], torch.zeros(2, dtype=torch.float64)))
            self.assertFalse(pred["cop_valid"][0])
            self.assertTrue((pred["tangent_power"] <= 0).all())
            self.assertTrue((pred["damping_power"] <= 1e-12).all())
            self.assertTrue((torch.linalg.vector_norm(pred["tangent_force"], dim=-1) <= 1.5 * pred["normal_force"]).all())
            self.assertTrue((pred["cop"].abs()[:, 0] <= .05).all())

    def test_rollout_gradients_and_switch(self):
        for kind in ("analytic", "contact_net"):
            model = ContactModel(kind, {"mass": .8, "taxel_positions": [[-.05, 0], [.05, 0]]})
            result = finite_difference_check(model, [0, 0, -.0008, .15, .03, -.02], [[0, 0, 0]] * 12, .001)
            self.assertEqual(result["status"], "PASS")
            # Gap=0 is an explicit non-smooth boundary. Probe both sides,
            # asserting physical behavior rather than claiming differentiability.
            for gap in (-1e-7, 0., 1e-7):
                force = model.predict({"gap": [gap, gap], "velocity": [0, 0, -.02],
                                       "taxel_positions": [[-.05, 0], [.05, 0]]})["normal_force"]
                self.assertTrue(torch.isfinite(force))
                self.assertEqual(float(force) == 0, gap >= 0)

    def test_audit_rejects_leakage_units_missing_labels(self):
        data = synthetic_trials(seed=17, count=2, held_out_count=1)
        self.assertEqual(audit_dataset(data)["status"], "PASS")
        bad = copy.deepcopy(data)
        bad.trials[-1].metadata["session_id"] = bad.trials[0].metadata["session_id"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            audit_dataset(bad)
        bad = copy.deepcopy(data)
        bad.trials[0].units["time"] = "ms"
        with self.assertRaisesRegex(ValueError, "SI units"):
            audit_dataset(bad)
        bad = copy.deepcopy(data)
        bad.trials[0].tangent_force = None
        with self.assertRaisesRegex(ValueError, "missing labels"):
            audit_dataset(bad)
        with self.assertRaisesRegex(ValueError, "manifest task"):
            audit_dataset(data, "trajectory_fit")
        bad = copy.deepcopy(data)
        bad.trials[-1] = copy.deepcopy(bad.trials[0])
        bad.trials[-1].trial_id = data.trials[-1].trial_id
        bad.trials[-1].metadata["session_id"] = "renamed"
        with self.assertRaisesRegex(ValueError, "duplicate trial"):
            audit_dataset(bad)

    def test_data_and_checkpoint_roundtrip(self):
        import json
        from dataclasses import asdict
        from skate_sim.physics.contact import MODEL_SCHEMA
        data = synthetic_trials(seed=9, count=1, held_out_count=1, task="trajectory_fit")
        self.assertEqual(audit_dataset(data, "trajectory_fit")["status"], "PASS")
        no_force_labels = copy.deepcopy(data)
        for trial in no_force_labels.trials:
            trial.taxel_force = trial.tangent_force = None
        self.assertEqual(audit_dataset(no_force_labels, "trajectory_fit")["status"], "PASS")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "data.json").write_text(json.dumps({"schema": CONTACT_SCHEMA, **asdict(data)}))
            self.assertEqual(load_trials(path / "data.json").splits, data.splits)
            model = ContactModel("contact_net")
            torch.save({"schema": MODEL_SCHEMA, "model": model.name, "config": model.config,
                        "state_dict": model.module.state_dict()}, path / "model.pt")
            restored = load_contact_model(path / "model.pt")
            self.assertEqual(model.parameter_values(), restored.parameter_values())
            for name, value in model.module.state_dict().items():
                self.assertTrue(torch.equal(value, restored.module.state_dict()[name]))

    def test_free_flight_and_ood(self):
        model = ContactModel("contact_net", {"mass": .8, "taxel_positions": [[-.05, 0], [.05, 0]]})
        with torch.no_grad():
            states = model.rollout([0, 0, 1, .3, 0, 0], [[0, 0, 0]] * 100, .001)
        self.assertAlmostEqual(float(states[-1, 5]), -.981, places=10)
        self.assertAlmostEqual(float(states[-1, 0]), .03, places=10)
        pred = model.predict({"gap": [-.1, .1], "velocity": [10., -10., -5.],
                              "taxel_positions": [[-.05, 0], [.05, 0]]})
        self.assertTrue(torch.isfinite(pred["normal_force"]))
        self.assertLessEqual(float(pred["tangent_power"]), 0)
        self.assertEqual(float(pred["taxel_force"][1]), 0)
        with torch.no_grad():
            slowing = model.rollout([0, 0, -.0008, .01, 0, 0], [[0, 0, 0]] * 100, .001)
        self.assertTrue((slowing[:, 3] >= 0).all())
        self.assertTrue((torch.diff(slowing[:, 3]) <= 1e-12).all())

    def test_forward_gradients_all_parameters(self):
        model = ContactModel("analytic")
        features = {"gap": [[-.002, -.001]], "velocity": [[.3, -.1, -.02]],
                    "taxel_positions": [[-.05, 0], [.05, 0]]}
        def objective():
            prediction = model.predict(features)
            return prediction["normal_force"].sum() + prediction["tangent_force"].sum()
        gradient = torch.autograd.grad(objective(), model.module.raw)[0]
        original = model.module.raw.detach().clone()
        for index in range(3):
            self.assertGreater(abs(float(gradient[index])), 1e-6)
            for epsilon in (1e-3, 1e-4, 1e-5):
                values = []
                for sign in (-1, 1):
                    with torch.no_grad():
                        model.module.raw.copy_(original)
                        model.module.raw[index] += sign * epsilon
                        values.append(float(objective()))
                self.assertAlmostEqual((values[1] - values[0]) / (2 * epsilon), float(gradient[index]), delta=1e-4)
        with torch.no_grad():
            model.module.raw.copy_(original)


if __name__ == "__main__":
    unittest.main()
