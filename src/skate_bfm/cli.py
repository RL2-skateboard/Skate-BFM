from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from skate_bfm.runner import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Skate-BFM baseline smoke test.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(asdict(run_smoke(args.steps, args.seed)), indent=2))


if __name__ == "__main__":
    main()

