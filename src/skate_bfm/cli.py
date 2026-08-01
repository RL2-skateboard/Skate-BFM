from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from skate_bfm.runner import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Skate-BFM baseline smoke test.")
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Control steps to run; use 0 with --viewer to run until the window closes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewer", action="store_true", help="Open the interactive MuJoCo viewer.")
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Do not pace viewer execution to the control timestep.",
    )
    parser.add_argument(
        "--action-gain",
        type=float,
        default=0.0,
        help="Scale BFM0 actions before HUSKY mapping; the safe default is 0.0.",
    )
    args = parser.parse_args()
    summary = run_smoke(
        args.steps,
        args.seed,
        viewer=args.viewer,
        realtime=False if args.no_realtime else None,
        action_gain=args.action_gain,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
