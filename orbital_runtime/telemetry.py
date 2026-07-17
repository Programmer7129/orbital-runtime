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

Non-finite floats are written as the JSON STRINGS "NaN"/"Infinity"/
"-Infinity" (see `_encode_float`), because `json.dumps` would otherwise emit
them as bare `NaN`/`Infinity` literals, which are not JSON -- Python's
`json.loads` accepts them, but `JSON.parse` does not, and the dashboard is
JavaScript. This is not hypothetical: a dead run's `run_end` carries
`final_loss: NaN` and `final_val_loss: Infinity`, so the one record that
explains the death is exactly the record a browser could not read.

`read_events` reverses the encoding, so Python-side consumers see the same
floats they always did; only the bytes on disk changed.
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

# How non-finite floats cross the JSON boundary. `json.dumps` spells these
# as bare `NaN`/`Infinity`/`-Infinity`, which no conformant parser accepts;
# these strings are the same tokens quoted, so the intent survives a
# round-trip through any JSON reader in any language.
NONFINITE_ENCODINGS: dict[str, float] = {
    "NaN": float("nan"),
    "Infinity": float("inf"),
    "-Infinity": float("-inf"),
}


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
        # allow_nan=False turns "we wrote a non-JSON token" from a silent
        # corruption of the log into a loud failure at the write. If a float
        # ever reaches dumps unencoded, this raises instead of producing a
        # file the dashboard cannot parse.
        self._fh.write(
            json.dumps(_encode(rec), default=_jsonable, allow_nan=False) + "\n"
        )
        # The returned record keeps its real floats; callers (and tests) that
        # inspect an emitted record should see what happened, not the wire
        # spelling of it.
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


def _encode_float(x: float) -> float | str:
    """A JSON-safe spelling of a float. Finite values pass through."""
    if x != x:
        return "NaN"
    if x == float("inf"):
        return "Infinity"
    if x == float("-inf"):
        return "-Infinity"
    return x


def _encode(o: Any) -> Any:
    """Recursively replace non-finite floats with their string encodings.

    `json.dumps(default=...)` cannot do this: `default` is only consulted for
    types json does not know, and it knows float perfectly well -- it just
    spells the non-finite ones wrong.
    """
    if isinstance(o, float):
        return _encode_float(o)
    if isinstance(o, dict):
        return {k: _encode(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_encode(v) for v in o]
    return o


def _decode(o: Any) -> Any:
    """Reverse `_encode`.

    Caveat: a string whose genuine value is "NaN"/"Infinity"/"-Infinity"
    would be revived as a float. No string field in the schema can take those
    values (`kind`, `death_reason`, `target_kind`, `dtype`, `device`,
    `trigger` and friends are all drawn from fixed vocabularies; `name` is a
    parameter name), and `test_a_string_field_that_looks_like_a_float_is_the_
    known_limit` pins the boundary so it is a documented limit rather than a
    lurking surprise.
    """
    if isinstance(o, str):
        if o in NONFINITE_ENCODINGS:
            return NONFINITE_ENCODINGS[o]
        return o
    if isinstance(o, dict):
        return {k: _decode(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_decode(v) for v in o]
    return o


def _jsonable(o: Any) -> Any:
    """Last-resort encoder: numpy scalars, tensors, Paths.

    Runs *after* `_encode`, so a non-finite hiding inside a tensor or numpy
    scalar (which `_encode` cannot see through) is encoded here instead of
    reaching `dumps` and tripping `allow_nan=False`.
    """
    if hasattr(o, "item"):  # numpy scalar / 0-dim tensor
        try:
            return _encode(o.item())
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

    Non-finite floats are decoded back from their string encodings, so what
    comes out is what went in.
    """
    events: list[dict[str, Any]] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = _decode(json.loads(line))
            if strip_wall:
                for field_name in WALL_CLOCK_FIELDS:
                    rec.pop(field_name, None)
            events.append(rec)
    return events
