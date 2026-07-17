"""JSONL event log: the single source of truth for what happened in a run.

Consumed by the dashboard (M4), the overhead benchmark, and the tests. One
JSON object per line, appended as events occur, flushed every write so a
run that dies from an injected fault still leaves a complete log up to the
moment of death -- which is exactly the run we most need to inspect.

Event schema (common fields on every record):

    seq     monotonic counter, unique within a run
    kind    event type (see EVENT_* below)
    step    training step, or null before training starts
    t_sim   simulation time in seconds (orbit clock), or null
    wall    wall-clock seconds since run start

plus event-specific fields. Wall-clock readings are the ONLY nondeterministic
content; they are excluded from determinism comparisons (see
`read_events(strip_wall=True)` and `WALL_CLOCK_FIELDS`).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Lifecycle
EVENT_RUN_START = "run_start"
EVENT_RUN_END = "run_end"
EVENT_STEP = "step"
# Injection (M1)
EVENT_FLIP = "flip"  # a memory bit flip landed
EVENT_ACTIVATION = "activation"  # an activation element was corrupted
EVENT_SEFI = "sefi"  # simulated hang/crash
EVENT_XID = "xid"  # synthetic ECC/Xid report
# Detection (M2)
EVENT_DETECT = "detect"
# Recovery (M3)
EVENT_CHECKPOINT = "checkpoint"
EVENT_ROLLBACK = "rollback"

# Every field carrying a wall-clock reading. These are the only
# nondeterministic content in a log, so determinism checks project them out.
# Keep this list complete: a timing field that is missing here would make
# `strip_wall=True` silently fail to prove determinism.
WALL_CLOCK_FIELDS = frozenset({"wall", "wall_s"})


@dataclass
class Telemetry:
    """Append-only JSONL writer."""

    path: Path
    run_id: str
    tag: str = ""
    _fh: Any = field(default=None, init=False, repr=False)
    _seq: int = field(default=0, init=False, repr=False)
    _t0: float = field(default=0.0, init=False, repr=False)
    counts: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", buffering=1)  # line-buffered
        self._t0 = time.perf_counter()

    def emit(
        self,
        kind: str,
        *,
        step: int | None = None,
        t_sim: float | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "seq": self._seq,
            "kind": kind,
            "step": step,
            "t_sim": t_sim,
            "wall": round(time.perf_counter() - self._t0, 6),
        }
        rec.update(fields)
        self._seq += 1
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self._fh.write(json.dumps(rec, default=_jsonable) + "\n")
        return rec

    def count(self, kind: str) -> int:
        return self.counts.get(kind, 0)

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> Telemetry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _jsonable(o: Any) -> Any:
    """Last-resort encoder: numpy scalars, tensors, Paths."""
    if hasattr(o, "item"):  # numpy scalar / 0-dim tensor
        try:
            return o.item()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_events(
    path: str | Path, *, strip_wall: bool = False
) -> list[dict[str, Any]]:
    """Read a JSONL log back.

    `strip_wall=True` drops every wall-clock field, leaving only the
    deterministic content -- two runs with the same seed must produce
    identical event streams under this projection.
    """
    events: list[dict[str, Any]] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if strip_wall:
                for field_name in WALL_CLOCK_FIELDS:
                    rec.pop(field_name, None)
            events.append(rec)
    return events
