"""JSONL telemetry: the log the dashboard and the tests both read."""

from __future__ import annotations

import json

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
        assert 0 <= f["bit"] < 32
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
