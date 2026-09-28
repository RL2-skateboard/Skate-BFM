from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class Recorder:
    """Line-oriented record format; every event is independently readable."""

    def __init__(self, path: str | None, *, backend: str, scene: str, seed: int):
        self.path = Path(path) if path else None
        self.file = None
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            self.file = (self.path / "events.jsonl").open("w", encoding="utf-8")
            self.write("meta", backend=backend, scene=scene, seed=seed,
                       wall_time=time.time(), platform=platform.platform())

    def write(self, event: str, **data: Any) -> None:
        if self.file:
            self.file.write(json.dumps({"event": event, "time": time.time(), **_jsonable(data)}) + "\n")
            self.file.flush()

    def close(self) -> None:
        if self.file:
            self.file.close()
