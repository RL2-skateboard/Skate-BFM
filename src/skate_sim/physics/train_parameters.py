"""Offline contact identification; interactive comparison stays in play.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="skate-sim offline contact parameter training")
    p.add_argument("--data", required=True, help="audited contact JSON, or explicit 'synthetic'")
    p.add_argument("--task", choices=("force_fit", "trajectory_fit"), default="force_fit")
    p.add_argument("--model", choices=("analytic", "contact_net"), default="analytic")
    p.add_argument("--init", help="analytic model.pt baseline required by contact_net")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--learning-rate", type=float, default=.03)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output", required=True, help="new run directory (existing artifacts are never overwritten)")
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        # Help works with only the standard library; native play never imports torch.
        from skate_sim.physics.contact import (ContactFitConfig, fit_contact, load_trials,
                                                save_contact_run, synthetic_trials)
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("output directory is not empty; choose a new run directory")
        dataset = (synthetic_trials(seed=args.seed, task=args.task) if args.data == "synthetic"
                   else load_trials(args.data))
        result = fit_contact(dataset, ContactFitConfig(task=args.task, model=args.model,
                             epochs=args.epochs, learning_rate=args.learning_rate,
                             seed=args.seed, init=args.init))
        save_contact_run(output, dataset, result)
        report = result["report"]
        print(json.dumps({"status": report["status"], "origin": report["origin"],
                          "checks": report["checks"], "parameters": report["parameters"],
                          "test_loss": report["metrics"]["test"]["loss"],
                          "output": str(output)}, indent=2))
        return 0 if report["status"] == "PASS" else 1
    except (ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
