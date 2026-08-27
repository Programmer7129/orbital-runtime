"""JSONL telemetry: the log the dashboard and the tests both read."""

from __future__ import annotations

import json
import math

import pytest

from orbital_runtime.inject.injector import RadiationEnvironment
from orbital_runtime.inject.memory import MemoryInjector
from orbital_runtime.orbit.flux import FluxModel
from orbital_runtime.orbit.track import OrbitTrack
from orbital_runtime.telemetry import (
    EVENT_FLIP,
    EVENT_RUN_END,
    EVENT_RUN_START,
    WALL_CLOCK_FIELDS,
    Telemetry,
    read_events,
)
from orbital_runtime.train import TrainConfig, train


def test_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "t.jsonl"
    with Telemetry(path=path, run_id="r") as t:
        t.emit("alpha", step=1, t_sim=2.0, x=1)
        t.emit("beta", step=2, t_sim=3.0, y="two")

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["kind"] == "alpha"
    assert json.loads(lines[1])["y"] == "two"


def test_seq_is_monotonic_and_counts_accumulate(tmp_path):
    with Telemetry(path=tmp_path / "t.jsonl", run_id="r") as t:
        for i in range(5):
            rec = t.emit("flip", step=i)
            assert rec["seq"] == i
        assert t.count("flip") == 5
        assert t.count("nope") == 0


def test_log_survives_a_crash_mid_run(tmp_path):
    """Line-buffered on purpose.

    The run we most need to inspect is the one that died. If the log were
    block-buffered, a process killed by a SEFI would lose the events
    explaining its own death.
    """
    path = tmp_path / "t.jsonl"
    t = Telemetry(path=path, run_id="r")
    t.emit("flip", step=1)
    t.emit("flip", step=2)
    # Deliberately do NOT close: simulate a hard death.
    assert len(read_events(path)) == 2


def test_read_events_can_strip_every_wall_clock_field(tmp_path):
    """A timing field missed here would make determinism checks vacuous."""
    path = tmp_path / "t.jsonl"
    with Telemetry(path=path, run_id="r") as t:
        t.emit("step", step=0, loss=1.0)
        t.emit("run_end", step=1, wall_s=0.5)

    raw = read_events(path)
    assert "wall" in raw[0]
    assert "wall_s" in raw[1]

    stripped = read_events(path, strip_wall=True)
    assert not any(f in rec for rec in stripped for f in WALL_CLOCK_FIELDS)
    assert stripped[0]["loss"] == 1.0  # non-timing content survives


def test_common_fields_present_on_every_record(tmp_path):
    path = tmp_path / "t.jsonl"
    with Telemetry(path=path, run_id="r") as t:
        t.emit("thing")
    rec = read_events(path)[0]
    assert set(rec) >= {"seq", "kind", "step", "t_sim", "wall"}
    assert rec["step"] is None and rec["t_sim"] is None  # explicit nulls


def test_numpy_and_tensor_values_are_serialisable(tmp_path):
    import numpy as np
    import torch

    path = tmp_path / "t.jsonl"
    with Telemetry(path=path, run_id="r") as t:
        t.emit("x", a=np.float32(1.5), b=np.int64(3), c=torch.tensor(2.5))
    rec = read_events(path)[0]
    assert rec["a"] == 1.5 and rec["b"] == 3 and rec["c"] == 2.5


# --------------------------------------------------------------------- #
# JSON conformance: the dashboard is JavaScript, not Python
# --------------------------------------------------------------------- #


def _strict_load(line: str):
    """Parse a line the way a non-Python reader would.

    `json.loads` accepts bare NaN/Infinity as an extension; `JSON.parse` and
    the JSON spec do not. `parse_constant` is the hook that fires on exactly
    those tokens, so this rejects what a browser would reject.
    """

    def reject(token):
        raise ValueError(f"not valid JSON: bare {token} literal")

    return json.loads(line, parse_constant=reject)


def test_non_finite_floats_are_written_as_valid_json(tmp_path):
    """A dead run's run_end carries NaN and inf.

    Written as bare literals -- json.dumps's default -- that record is
    unparseable by JSON.parse, so the dashboard would fail to read the one
    event that explains the death. Encoded as strings, every reader can.
    """
    path = tmp_path / "t.jsonl"
    with Telemetry(path=path, run_id="r") as t:
        t.emit(
            EVENT_RUN_END,
            step=7,
            died=True,
            final_loss=float("nan"),
            final_val_loss=float("inf"),
            worst=float("-inf"),
            fine=2.5,
        )

    raw = _strict_load(path.read_text().strip())
    assert raw["final_loss"] == "NaN"
    assert raw["final_val_loss"] == "Infinity"
    assert raw["worst"] == "-Infinity"
    assert raw["fine"] == 2.5  # finite floats stay numbers


def test_read_events_restores_the_real_floats(tmp_path):
    """The encoding is a wire format, not a change of meaning."""
    path = tmp_path / "t.jsonl"
    with Telemetry(path=path, run_id="r") as t:
        t.emit("x", loss=float("nan"), val=float("inf"), low=float("-inf"), ok=1.25)

    rec = read_events(path)[0]
    assert math.isnan(rec["loss"])
    assert rec["val"] == float("inf")
    assert rec["low"] == float("-inf")
    assert rec["ok"] == 1.25


def test_non_finite_inside_a_tensor_is_encoded_too(tmp_path):
    """`_encode` cannot see through a tensor; `_jsonable` must finish the job.

    Without that, a NaN tensor reaches dumps unencoded and allow_nan=False
    raises -- a regression on a path that used to (wrongly) work.
    """
    import numpy as np
    import torch

    path = tmp_path / "t.jsonl"
    with Telemetry(path=path, run_id="r") as t:
        t.emit("x", a=torch.tensor(float("nan")), b=np.float32("inf"))

    raw = _strict_load(path.read_text().strip())
    assert raw["a"] == "NaN" and raw["b"] == "Infinity"
    rec = read_events(path)[0]
    assert math.isnan(rec["a"]) and rec["b"] == float("inf")


def test_nested_non_finite_is_encoded(tmp_path):
    """run_end splats **stats in, so nesting is reachable."""
    path = tmp_path / "t.jsonl"
    with Telemetry(path=path, run_id="r") as t:
        t.emit("x", d={"inner": float("nan")}, lst=[1.0, float("inf")])

    raw = _strict_load(path.read_text().strip())
    assert raw["d"]["inner"] == "NaN"
    assert raw["lst"] == [1.0, "Infinity"]


def test_emit_returns_the_real_float_not_the_wire_spelling(tmp_path):
    with Telemetry(path=tmp_path / "t.jsonl", run_id="r") as t:
        rec = t.emit("x", loss=float("nan"))
    assert math.isnan(rec["loss"])


def test_a_string_field_that_looks_like_a_float_is_the_known_limit(tmp_path):
    """Documents the one ambiguity the string encoding buys.

    No field in the event schema can legitimately hold these values, so this
    pins the boundary rather than guarding a live hazard. If a future field
    could, it needs a different encoding -- not a tweak to this test.
    """
    path = tmp_path / "t.jsonl"
    with Telemetry(path=path, run_id="r") as t:
        t.emit("x", note="NaN", real="fine")

    rec = read_events(path)[0]
    assert math.isnan(rec["note"])  # the string came back as a float
    assert rec["real"] == "fine"


def test_a_whole_dead_run_log_is_valid_json(tiny_workload, tmp_path):
    """End to end: every line of a log from a run that died parses strictly."""
    path = tmp_path / "run.jsonl"
    telemetry = Telemetry(path=path, run_id="d")

    w = tiny_workload(seed=5)
    bits = MemoryInjector(w.model, w.optimizer).static_resident_bits()
    flux = FluxModel(bits_resident=bits, base_rate_upsets_per_bit_day=5e-4)
    env = RadiationEnvironment(
        w.model, w.optimizer, flux=flux, seed=5, n_steps=120, orbits=2.0, telemetry=telemetry
    )
    result = train(w, cfg=TrainConfig(steps=120), env=env, telemetry=telemetry)
    telemetry.close()
    assert result.died  # the log we care about is the one that died

    lines = path.read_text().strip().split("\n")
    for i, line in enumerate(lines):
        try:
            _strict_load(line)
        except ValueError as e:
            raise AssertionError(f"line {i} ({json.loads(line)['kind']}): {e}") from e


def test_close_is_idempotent(tmp_path):
    t = Telemetry(path=tmp_path / "t.jsonl", run_id="r")
    t.close()
    t.close()


def test_creates_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "t.jsonl"
    with Telemetry(path=path, run_id="r") as t:
        t.emit("x")
    assert path.exists()


# --------------------------------------------------------------------- #
# Integration: a real run's log
# --------------------------------------------------------------------- #


def test_run_log_tells_the_whole_story(tiny_workload, tmp_path):
    """The dashboard reads only this file, so it must be self-sufficient."""
    path = tmp_path / "run.jsonl"
    telemetry = Telemetry(path=path, run_id="demo", tag="demo")

    w = tiny_workload(seed=1)
    bits = MemoryInjector(w.model, w.optimizer).static_resident_bits()
    flux = FluxModel(bits_resident=bits, base_rate_upsets_per_bit_day=5e-4)
    env = RadiationEnvironment(
        w.model, w.optimizer, flux=flux, seed=1, n_steps=120, orbits=2.0, telemetry=telemetry
    )
    result = train(w, cfg=TrainConfig(steps=120), env=env, telemetry=telemetry)
    telemetry.close()

    events = read_events(path)
    kinds = [e["kind"] for e in events]
    assert kinds[0] == EVENT_RUN_START
    assert kinds[-1] == EVENT_RUN_END
    assert EVENT_FLIP in kinds

    start = events[0]
    assert start["scheduled_upsets"] == env.scheduled_upsets
    assert start["irradiated"] is True
    assert start["protected"] is False

    flips = [e for e in events if e["kind"] == EVENT_FLIP]
    assert len(flips) == env.stats.flips
    for f in flips:
        assert f["value_before"] != f["value_after"]
        # bit == -1 is the sentinel for a DATAPATH fault: nullification or a
        # forced special value overwrote the element rather than toggling a
        # bit. Those mechanisms did not exist when this was written.
        assert f["bit"] == -1 or 0 <= f["bit"] < 32
        assert f["target_kind"] in ("param", "optimizer")
        assert isinstance(f["in_saa"], bool)
        assert f["t_sim"] is not None

    end = events[-1]
    assert end["died"] == result.died
    assert end["flips"] == env.stats.flips


def test_run_end_is_written_even_when_the_run_dies(tiny_workload, tmp_path):
    path = tmp_path / "run.jsonl"
    telemetry = Telemetry(path=path, run_id="d")

    w = tiny_workload(seed=5)
    bits = MemoryInjector(w.model, w.optimizer).static_resident_bits()
    flux = FluxModel(bits_resident=bits, base_rate_upsets_per_bit_day=5e-4)
    env = RadiationEnvironment(
        w.model, w.optimizer, flux=flux, seed=5, n_steps=120, orbits=2.0, telemetry=telemetry
    )
    result = train(w, cfg=TrainConfig(steps=120), env=env, telemetry=telemetry)
    telemetry.close()
    assert result.died

    end = read_events(path)[-1]
    assert end["kind"] == EVENT_RUN_END
    assert end["died"] is True
    assert end["death_reason"] == "nan_loss"


def test_two_seeded_runs_produce_identical_logs(tiny_workload, tmp_path):
    """Determinism, asserted on the artifact the demo actually reads."""

    def run(name: str):
        path = tmp_path / f"{name}.jsonl"
        telemetry = Telemetry(path=path, run_id=name)
        w = tiny_workload(seed=3)
        bits = MemoryInjector(w.model, w.optimizer).static_resident_bits()
        flux = FluxModel(bits_resident=bits, base_rate_upsets_per_bit_day=2e-4)
        env = RadiationEnvironment(
            w.model, w.optimizer, flux=flux, seed=3, n_steps=60, orbits=2.0, telemetry=telemetry
        )
        train(w, cfg=TrainConfig(steps=60), env=env, telemetry=telemetry)
        telemetry.close()
        return read_events(path, strip_wall=True)

    assert run("a") == run("b")


def test_strip_wall_leaves_no_nondeterministic_field(tiny_workload, tmp_path):
    """WALL_CLOCK_FIELDS completeness (item 5).

    A PROTECTED run emits `checkpoint_wall_s` on run_end (via the recovery
    stats) -- a wall-clock field that was NOT in WALL_CLOCK_FIELDS, so
    `strip_wall=True` left it in and two identical seeded runs disagreed on
    exactly that one event. This is the protected analogue of
    test_two_seeded_runs_produce_identical_logs (which is unprotected and so
    never exercised checkpoint_wall_s).
    """
    from orbital_runtime.ckpt.policy import CheckpointPolicy
    from orbital_runtime.ckpt.recover import RecoveryOrchestrator
    from orbital_runtime.ckpt.saver import CheckpointSaver
    from orbital_runtime.detect import Detector, GuardTier
    from orbital_runtime.detect.abft import AbftTier
    from orbital_runtime.rng import STREAM_ABFT, stream

    def run(name: str):
        path = tmp_path / f"{name}.jsonl"
        telemetry = Telemetry(path=path, run_id=name)
        w = tiny_workload(seed=3)
        bits = MemoryInjector(w.model, w.optimizer).static_resident_bits()
        flux = FluxModel(bits_resident=bits, base_rate_upsets_per_bit_day=5e-4)
        env = RadiationEnvironment(
            w.model, w.optimizer, flux=flux, seed=3, n_steps=100, orbits=2.0,
            telemetry=telemetry,
        )
        abft = AbftTier(
            w.model, base_sample_rate=0.1, saa_sample_rate=1.0, adaptive=True,
            rng=stream(3, STREAM_ABFT),
        ).attach()
        detector = Detector(guards=GuardTier(), abft=abft)
        recovery = RecoveryOrchestrator(
            saver=CheckpointSaver(
                w.model, w.optimizer, directory=tmp_path / f"ck_{name}", use_async=False
            ),
            policy=CheckpointPolicy(track=OrbitTrack(), base_interval=25, saa_interval=8),
            env=env,
            detector=detector,
            telemetry=telemetry,
        )
        train(
            w, cfg=TrainConfig(steps=100), env=env, telemetry=telemetry,
            detector=detector, recovery=recovery,
        )
        telemetry.close()
        raw = read_events(path)
        run_end = next(e for e in raw if e["kind"] == "run_end")
        # The field that exposed the bug must actually be present, or this test
        # would pass vacuously.
        assert "checkpoint_wall_s" in run_end
        return read_events(path, strip_wall=True)

    a, b = run("a"), run("b")
    assert a == b
    # And no wall-clock field of any kind survives the projection.
    assert not any(f in rec for rec in a for f in WALL_CLOCK_FIELDS)
