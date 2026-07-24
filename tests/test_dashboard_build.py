"""The demo dashboard is only as honest as the bundle it reads.

`demo/dashboard/build.py` reduces the three telemetry logs into the JS the
page loads. These tests pin the reductions the dashboard depends on:

* non-finite floats (a dead run's `final_loss`/`final_val_loss`) become JSON
  `null`, never a bare `NaN` that `JSON.parse` would reject;
* the protected loss curve is the FINAL loss per step, so replays collapse to
  what actually survived, while `progress` keeps the emission order (with its
  backward rollback jumps) intact;
* every headline number in `summary` comes from a `run_end` record;
* orbit geometry is present and phase-gated, matching `OrbitTrack`.

They build the logs with the real `Telemetry` writer so the read path is the
same one the tests and the dashboard use.
"""

from __future__ import annotations

import json

from demo.dashboard import build as dash
from orbital_runtime.orbit.track import OrbitTrack
from orbital_runtime.telemetry import (
    EVENT_CHECKPOINT,
    EVENT_FLIP,
    EVENT_ROLLBACK,
    EVENT_RUN_END,
    EVENT_RUN_START,
    EVENT_STEP,
    Telemetry,
)


def _write_baseline(path):
    with Telemetry(path=path, run_id="baseline-s1337") as t:
        t.emit(EVENT_RUN_START, step=0, protected=False, irradiated=False,
               device="cpu", steps=3, scheduled_upsets=0)
        for s, loss in enumerate([4.0, 3.0, 2.5]):
            t.emit(EVENT_STEP, step=s, t_sim=float(s), loss=loss, in_saa=False)
        t.emit(EVENT_RUN_END, step=3, t_sim=3.0, died=False, death_reason=None,
               final_loss=2.5, final_val_loss=2.43, steps_executed=3, wall_s=1.0,
               flips=0, flips_in_saa=0)


def _write_unprotected(path):
    """A run that dies: last step has no finite loss, run_end carries NaN/inf."""
    with Telemetry(path=path, run_id="unprotected-s1337") as t:
        t.emit(EVENT_RUN_START, step=0, protected=False, irradiated=True,
               device="cpu", steps=3, scheduled_upsets=5)
        t.emit(EVENT_STEP, step=0, t_sim=0.0, loss=4.0, in_saa=False)
        t.emit(EVENT_FLIP, step=1, t_sim=1.0, in_saa=True, bit=30,
               target_kind="param", nonfinite=False)
        t.emit(EVENT_STEP, step=1, t_sim=1.0, loss=3.0, in_saa=True)
        # the fatal step: loss is NaN
        t.emit(EVENT_STEP, step=2, t_sim=2.0, loss=float("nan"), in_saa=True)
        t.emit(EVENT_RUN_END, step=2, t_sim=2.0, died=True, death_reason="nan_loss",
               final_loss=float("nan"), final_val_loss=float("inf"),
               steps_executed=3, wall_s=1.0, flips=1, flips_in_saa=1)


def _write_protected(path):
    """A run that rolls back: step 1 is replayed, so it appears twice."""
    with Telemetry(path=path, run_id="protected-s1337") as t:
        t.emit(EVENT_RUN_START, step=0, protected=True, irradiated=True,
               device="cpu", steps=3, scheduled_upsets=5)
        t.emit(EVENT_CHECKPOINT, step=0, t_sim=0.0, reason="interval", slot=0)
        t.emit(EVENT_STEP, step=0, t_sim=0.0, loss=4.0, in_saa=False)
        t.emit(EVENT_STEP, step=1, t_sim=1.0, loss=3.0, in_saa=True)   # first attempt
        t.emit(EVENT_ROLLBACK, step=1, t_sim=1.0, restored_to=0, replayed_steps=1,
               trigger_tier="abft", certain=True, lag=1, best_effort=False)
        t.emit(EVENT_STEP, step=1, t_sim=2.0, loss=2.9, in_saa=True)   # replay: this one sticks
        t.emit(EVENT_STEP, step=2, t_sim=3.0, loss=2.5, in_saa=False)
        t.emit(EVENT_RUN_END, step=3, t_sim=3.0, died=False, death_reason=None,
               final_loss=2.5, final_val_loss=2.42, steps_executed=4, wall_s=2.0,
               flips=1, flips_in_saa=1, detections=1,
               detections_by_tier={"abft": 1}, rollbacks=1, replayed_steps=1,
               abft_sample_rate_actual=0.19)


def _bundle(tmp_path):
    _write_baseline(tmp_path / "baseline-s1337.jsonl")
    _write_unprotected(tmp_path / "unprotected-s1337.jsonl")
    _write_protected(tmp_path / "protected-s1337.jsonl")
    return dash.build(tmp_path, seed=1337, generated_utc="")


def test_bundle_is_strict_json_with_no_nan(tmp_path):
    """The whole bundle must serialise under allow_nan=False (what the page needs)."""
    bundle = _bundle(tmp_path)
    # Would raise ValueError if any NaN/inf survived into the payload.
    text = json.dumps(bundle, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


def test_dead_run_nonfinite_becomes_null(tmp_path):
    bundle = _bundle(tmp_path)
    unprot = bundle["runs"]["unprotected"]
    assert unprot["died"] is True
    assert unprot["death_step"] == 2
    assert unprot["final_loss"] is None          # NaN -> null
    assert unprot["final_val_loss"] is None       # inf -> null
    # the fatal step's loss is null, so the dashboard stops the line there
    assert unprot["curve"][-1]["loss"] is None


def test_protected_curve_is_final_loss_per_step(tmp_path):
    """Step 1 was replayed; the curve keeps the loss that stuck (2.9), not 3.0."""
    bundle = _bundle(tmp_path)
    curve = bundle["runs"]["protected"]["curve"]
    by_step = {p["step"]: p["loss"] for p in curve}
    assert by_step[1] == 2.9                       # replay value, not the first attempt
    assert [p["step"] for p in curve] == [0, 1, 2]  # deduped + sorted


def test_protected_progress_keeps_the_rollback_jump(tmp_path):
    """progress is emission order, so it must contain the backward step jump."""
    bundle = _bundle(tmp_path)
    steps = [p["step"] for p in bundle["runs"]["protected"]["progress"]]
    assert steps == [0, 1, 1, 2]                   # the second 1 is the replay
    # somewhere a step number is followed by a not-greater one (the rollback)
    assert any(steps[i] <= steps[i - 1] for i in range(1, len(steps)))


def test_summary_comes_from_run_end(tmp_path):
    bundle = _bundle(tmp_path)
    s = bundle["summary"]
    assert s["upsets_injected"] == 1
    assert s["upsets_detected"] == 1
    assert s["rollbacks"] == 1
    assert s["replayed_steps"] == 1
    assert s["detections_by_tier"] == {"abft": 1}
    assert s["unprotected_death_step"] == 2
    assert s["baseline_val"] == 2.43
    assert s["protected_val"] == 2.42


def test_bundle_carries_no_wall_clock_field(tmp_path):
    """Item 5a: wall-clock is nondeterministic and must NOT be baked into the
    bundle, or the artifact is not byte-reproducible across reruns."""
    bundle = _bundle(tmp_path)
    s = bundle["summary"]
    for f in ("overhead_pct", "protected_wall_s", "baseline_wall_s"):
        assert f not in s, f"{f} leaked into the summary"
    for tag in ("baseline", "unprotected", "protected"):
        assert "wall_s" not in bundle["runs"][tag], f"wall_s leaked into runs[{tag}]"
    # The whole bundle text carries no wall-clock key at all.
    assert '"wall_s"' not in json.dumps(bundle)
    assert '"overhead_pct"' not in json.dumps(bundle)
    # Wall-clock overhead is still communicated -- as deterministic prose.
    assert bundle["meta"]["wall_overhead"]


def _serialize(bundle) -> str:
    """Exactly how build.main writes the payload -- the bytes that ship."""
    return json.dumps(bundle, indent=1, allow_nan=False)


def test_two_runs_differing_only_in_wall_clock_produce_identical_bytes(tmp_path):
    """Item 5: byte-exactness of the bundle.

    Wall-clock is the ONLY nondeterministic content of a run's log. Two runs
    that agree on everything except their wall readings must therefore reduce
    to a byte-identical bundle. This is the builder-level proof of what the
    empirical CPU rerun shows (two independent `--device cpu` demo runs hash
    to the same telemetry_data.js); on MPS the loss values themselves drift at
    the ULP, so byte-exactness is CPU-only and the determinism claim is worded
    as such (README / run_demo.sh).
    """

    def logs(root, wall):
        root.mkdir(parents=True, exist_ok=True)
        with Telemetry(path=root / "baseline-s1337.jsonl", run_id="baseline-s1337") as t:
            t.emit(EVENT_RUN_START, step=0, protected=False, irradiated=False,
                   device="cpu", steps=3, scheduled_upsets=0)
            for s, loss in enumerate([4.0, 3.0, 2.5]):
                t.emit(EVENT_STEP, step=s, t_sim=float(s), loss=loss, in_saa=False)
            t.emit(EVENT_RUN_END, step=3, t_sim=3.0, died=False, death_reason=None,
                   final_loss=2.5, final_val_loss=2.43, steps_executed=3,
                   wall_s=wall, flips=0, flips_in_saa=0)
        with Telemetry(path=root / "unprotected-s1337.jsonl", run_id="unprotected-s1337") as t:
            t.emit(EVENT_RUN_START, step=0, protected=False, irradiated=True,
                   device="cpu", steps=3, scheduled_upsets=5)
            t.emit(EVENT_STEP, step=0, t_sim=0.0, loss=4.0, in_saa=False)
            t.emit(EVENT_STEP, step=1, t_sim=1.0, loss=float("nan"), in_saa=True)
            t.emit(EVENT_RUN_END, step=1, t_sim=1.0, died=True, death_reason="nan_loss",
                   final_loss=float("nan"), final_val_loss=float("inf"),
                   steps_executed=2, wall_s=wall, flips=1, flips_in_saa=1)
        with Telemetry(path=root / "protected-s1337.jsonl", run_id="protected-s1337") as t:
            t.emit(EVENT_RUN_START, step=0, protected=True, irradiated=True,
                   device="cpu", steps=3, scheduled_upsets=5)
            t.emit(EVENT_CHECKPOINT, step=0, t_sim=0.0, reason="interval", slot=0)
            t.emit(EVENT_STEP, step=0, t_sim=0.0, loss=4.0, in_saa=False)
            t.emit(EVENT_STEP, step=1, t_sim=1.0, loss=2.5, in_saa=True)
            t.emit(EVENT_RUN_END, step=2, t_sim=2.0, died=False, death_reason=None,
                   final_loss=2.5, final_val_loss=2.42, steps_executed=2,
                   # wall AND the summed checkpoint wall both vary; both are
                   # in WALL_CLOCK_FIELDS, so neither may reach the bundle.
                   wall_s=wall, checkpoint_wall_s=wall * 0.5, flips=1, flips_in_saa=1,
                   detections=1, detections_by_tier={"abft": 1}, rollbacks=1,
                   replayed_steps=1, abft_sample_rate_actual=0.19)

    logs(tmp_path / "run_a", wall=1.0)
    logs(tmp_path / "run_b", wall=999.0)
    a = _serialize(dash.build(tmp_path / "run_a", seed=1337, generated_utc=""))
    b = _serialize(dash.build(tmp_path / "run_b", seed=1337, generated_utc=""))
    assert a == b, "wall-clock leaked into the bundle -- it is not byte-reproducible"


def test_orbit_geometry_matches_the_real_track(tmp_path):
    bundle = _bundle(tmp_path)
    orbit = bundle["orbit"]
    track = OrbitTrack()
    assert orbit["period_s"] == track.period_s
    assert orbit["saa_start_phase"] == track.saa_start_phase
    assert orbit["saa_end_phase"] == track.saa_start_phase + track.saa_fraction
    assert len(orbit["path"]) > 0
    # SAA membership on the path is phase-gated exactly as the model gates it
    for p in orbit["path"]:
        assert p["in_saa"] == (orbit["saa_start_phase"] <= p["phase"] < orbit["saa_end_phase"])
