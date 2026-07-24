"""Bundle the three demo telemetry logs into one JS file the dashboard loads.

The dashboard is a self-contained local page (no server, no CDN, no `fetch`).
So the data crosses into it the only way a `file://` page can read reliably:
as a `<script src>`. This module reads `runs/<tag>-s<seed>.jsonl` with the
same `read_events` the tests use, reduces each run to exactly what the four
panels need, and writes `telemetry_data.js` as `window.TELEMETRY = {...}`.

Design choices that keep the dashboard honest:

* **Orbit geometry is computed by the real `OrbitTrack`, not reimplemented in
  JS.** SAA membership is phase-gated (see `orbit/track.py`); the ground track
  (lat/lon) is display-only and is labelled as such. Precomputing here means
  the page cannot silently drift from the model.
* **The protected loss curve is the final loss per progress-step.** A replayed
  step re-emits the same step number (this is truthful, not a bug); the curve
  a viewer reads as "did the model survive" is the successful pass at each
  step. Rollbacks are surfaced separately as their own markers, so the
  recovery story is shown, not deduped away.
* **Every number in `summary` comes straight from a `run_end` record** — the
  same source the CLI prints and the tests assert on. The dashboard computes
  no physics of its own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from orbital_runtime.orbit.track import OrbitTrack
from orbital_runtime.telemetry import read_events

# The canonical demo runs (see demo/run_demo.sh and the Makefile `demo` target).
DEFAULT_SEED = 1337
TAGS = ("baseline", "unprotected", "protected")


def _events(run_dir: Path, tag: str, seed: int) -> list[dict[str, Any]]:
    path = run_dir / f"{tag}-s{seed}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"missing telemetry log: {path}\n"
            f"Run demo/run_demo.sh (or `make demo`) first to produce the runs."
        )
    return read_events(path)


def _run_end(events: list[dict[str, Any]]) -> dict[str, Any]:
    for e in events:
        if e["kind"] == "run_end":
            return e
    raise SystemExit("telemetry log has no run_end record — was the run cut off?")


def _run_start(events: list[dict[str, Any]]) -> dict[str, Any]:
    for e in events:
        if e["kind"] == "run_start":
            return e
    raise SystemExit("telemetry log has no run_start record.")


def _num(x: Any) -> Any:
    """JSON-safe: non-finite floats become null so the page can test for them.

    `read_events` has already decoded the wire strings back to real floats, so
    a NaN/inf can be sitting in `final_loss`/`final_val_loss` for a dead run.
    `JSON.stringify` on the browser side would choke on those; null is the
    value JS actually tests with `x == null`.
    """
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    return x


def _progress_curve(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Final loss per progress-step, ordered by step.

    Replays re-emit a step number; the last emission for a step is the one
    that stuck. Keyed by step, so a protected run's curve is what actually
    survived, not the churn underneath it.
    """
    by_step: dict[int, dict[str, Any]] = {}
    for e in events:
        if e["kind"] != "step" or "loss" not in e:
            continue
        by_step[e["step"]] = {
            "step": e["step"],
            "loss": _num(e["loss"]),
            "t_sim": e.get("t_sim"),
            "in_saa": e.get("in_saa", False),
        }
    return [by_step[s] for s in sorted(by_step)]


def _flips(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "t_sim": e.get("t_sim"),
            "step": e.get("step"),
            "in_saa": e.get("in_saa", False),
            "bit": e.get("bit"),
            "target_kind": e.get("target_kind"),
            "nonfinite": e.get("nonfinite", False),
        }
        for e in events
        if e["kind"] == "flip"
    ]


def _rollbacks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "t_sim": e.get("t_sim"),
            "step": e.get("step"),
            "restored_to": e.get("restored_to"),
            "replayed_steps": e.get("replayed_steps"),
            "trigger_tier": e.get("trigger_tier"),
            "certain": e.get("certain"),
            "lag": e.get("lag"),
        }
        for e in events
        if e["kind"] == "rollback"
    ]


def _checkpoints(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"t_sim": e.get("t_sim"), "step": e.get("step"), "reason": e.get("reason")}
        for e in events
        if e["kind"] == "checkpoint"
    ]


def _reduce_run(events: list[dict[str, Any]]) -> dict[str, Any]:
    start, end = _run_start(events), _run_end(events)
    run = {
        "protected": start.get("protected", False),
        "irradiated": start.get("irradiated", False),
        "device": start.get("device"),
        "scheduled_upsets": start.get("scheduled_upsets", 0),
        "died": end.get("died", False),
        "death_reason": end.get("death_reason"),
        "death_step": end.get("step") if end.get("died") else None,
        "steps_requested": start.get("steps"),
        "steps_executed": end.get("steps_executed"),
        # NOTE: wall-clock is deliberately NOT copied into the bundle. It is the
        # only nondeterministic content of a run, and a bundle that carried it
        # would not be byte-reproducible across reruns -- breaking the demo's
        # determinism claim on MPS (hostile review, item 5). Wall-clock overhead
        # is a real measurement, surfaced as deterministic prose in meta
        # (wall_overhead), not as a live float baked into the artifact.
        "final_loss": _num(end.get("final_loss")),
        "final_val_loss": _num(end.get("final_val_loss")),
        "flips_total": end.get("flips", 0),
        "flips_in_saa": end.get("flips_in_saa", 0),
        "curve": _progress_curve(events),
    }
    if run["irradiated"]:
        run["flips"] = _flips(events)
    if run["protected"]:
        # Emission-order (t_sim, step, loss) of every step event, replays and
        # all. The dashboard walks this to place the "you are here" marker,
        # which JUMPS BACKWARD on a rollback -- the recovery shown, not deduped
        # (STATUS handoff §1: the repeated step numbers ARE the story).
        run["progress"] = [
            {"t_sim": e.get("t_sim"), "step": e["step"], "loss": _num(e["loss"])}
            for e in events
            if e["kind"] == "step" and "loss" in e
        ]
        run["rollbacks"] = _rollbacks(events)
        run["checkpoints"] = _checkpoints(events)
        run["detections"] = end.get("detections", 0)
        run["detections_by_tier"] = end.get("detections_by_tier", {})
        run["replayed_steps"] = end.get("replayed_steps", 0)
        run["abft_sample_rate_actual"] = _num(end.get("abft_sample_rate_actual"))
    return run


def _orbit_geometry(t_max: float, n_samples: int = 240) -> dict[str, Any]:
    """Display geometry from the real orbit model.

    The ring/phase view uses `phase`; the ground-track readout uses
    `ground_track`. Both come from `OrbitTrack` so the page never invents a
    second, drifting copy of the orbit.
    """
    track = OrbitTrack()
    path = []
    for i in range(n_samples + 1):
        t = t_max * i / n_samples
        lat, lon = track.ground_track(t)
        path.append(
            {
                "t": round(t, 1),
                "phase": round(track.phase(t), 5),
                "lat": round(lat, 2),
                "lon": round(lon, 2),
                "in_saa": track.in_saa(t),
            }
        )
    return {
        "period_s": track.period_s,
        "saa_duration_s": track.saa_duration_s,
        "saa_start_phase": track.saa_start_phase,
        "saa_end_phase": track.saa_start_phase + track.saa_fraction,
        "saa_fraction": track.saa_fraction,
        "inclination_deg": track.inclination_deg,
        "orbit_min": track.period_s / 60.0,
        "saa_min": track.saa_duration_s / 60.0,
        "path": path,
    }


# Defaults describe the laptop demo (demo/run_demo.sh). The M4b L4 build
# overrides these on the command line to describe the calibrated real-scale run.
DEFAULT_RATE_LABEL = "3e-6 upsets/bit-day (300x the calibrated flight band)"
DEFAULT_DETECTION_OVERHEAD = "+5.4% at scale (MPS, 10.7M params) -- see README"
DEFAULT_NOTE = (
    "Rate 3e-6 upsets/bit-day is 300x the calibrated flight band, "
    "compensating for this demo model's 7.8e7 resident bits vs an "
    "H100's 6.4e11. The band itself is asserted in tests/test_flux.py; "
    "M4b re-measures at real scale on a GPU."
)
# Wall-clock overhead is a real measurement but nondeterministic, so it is
# stated as prose (deterministic) rather than baked into the bundle as a live
# float. The laptop default describes the replay-dominated demo-rate figure.
DEFAULT_WALL_OVERHEAD = (
    "wall-clock overhead is replay-dominated at this elevated demo rate and "
    "varies run to run; the calibrated detection-only figure is the one to "
    "quote (see meta.detection_overhead and the README)."
)


def build(
    run_dir: Path,
    seed: int,
    generated_utc: str,
    *,
    rate_label: str = DEFAULT_RATE_LABEL,
    detection_overhead: str = DEFAULT_DETECTION_OVERHEAD,
    note: str = DEFAULT_NOTE,
    wall_overhead: str = DEFAULT_WALL_OVERHEAD,
) -> dict[str, Any]:
    runs = {tag: _reduce_run(_events(run_dir, tag, seed)) for tag in TAGS}

    base, unprot, prot = runs["baseline"], runs["unprotected"], runs["protected"]

    # Mission clock spans the longest run (the protected one flies past where
    # the unprotected run died). Everything animates against this. t_sim is the
    # deterministic orbit clock (executed-work map), never wall time.
    t_max = max(
        (p["t_sim"] or 0.0)
        for run in runs.values()
        for p in run["curve"]
    )

    summary = {
        "seed": seed,
        "upsets_injected": prot["flips_total"],
        "upsets_in_saa": prot["flips_in_saa"],
        "upsets_detected": prot.get("detections", 0),
        "detections_by_tier": prot.get("detections_by_tier", {}),
        "rollbacks": len(prot.get("rollbacks", [])),
        "replayed_steps": prot.get("replayed_steps", 0),
        "abft_sample_rate_pct": (
            round(100.0 * prot["abft_sample_rate_actual"], 1)
            if prot.get("abft_sample_rate_actual") is not None
            else None
        ),
        "baseline_val": base["final_val_loss"],
        "protected_val": prot["final_val_loss"],
        "unprotected_death_step": unprot["death_step"],
        "unprotected_death_reason": unprot["death_reason"],
        "steps_requested": base["steps_requested"],
        # No wall_s/overhead_pct here on purpose (item 5): they are
        # nondeterministic. The recovery story below (rollbacks, replayed_steps)
        # is deterministic and is the honest headline; wall-clock overhead is
        # carried as prose in meta.wall_overhead.
        "device": prot["device"],
    }

    return {
        "meta": {
            "generated_utc": generated_utc,
            "seed": seed,
            "rate_label": rate_label,
            "detection_overhead": detection_overhead,
            "wall_overhead": wall_overhead,
            "note": note,
        },
        "orbit": _orbit_geometry(t_max),
        "t_max": round(t_max, 1),
        "runs": runs,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bundle demo telemetry into telemetry_data.js")
    p.add_argument("--run-dir", type=Path, default=Path("runs"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "telemetry_data.js",
    )
    # Passed in (not read from the clock) so the build is deterministic and the
    # test suite can call build() without a wall-clock dependency.
    p.add_argument("--generated-utc", default="")
    # M4b: describe the actual run. Defaults describe the laptop demo; the L4
    # build overrides them to describe the calibrated real-scale mission.
    p.add_argument("--rate-label", default=DEFAULT_RATE_LABEL)
    p.add_argument("--detection-overhead", default=DEFAULT_DETECTION_OVERHEAD)
    p.add_argument("--note", default=DEFAULT_NOTE)
    p.add_argument("--wall-overhead", default=DEFAULT_WALL_OVERHEAD)
    args = p.parse_args(argv)

    bundle = build(
        args.run_dir,
        args.seed,
        args.generated_utc,
        rate_label=args.rate_label,
        detection_overhead=args.detection_overhead,
        note=args.note,
        wall_overhead=args.wall_overhead,
    )
    payload = json.dumps(bundle, indent=1, allow_nan=False)
    args.out.write_text(
        "// GENERATED by demo/dashboard/build.py from runs/*.jsonl — do not edit.\n"
        "// Regenerate with: demo/run_demo.sh  (or: python -m demo.dashboard.build)\n"
        f"window.TELEMETRY = {payload};\n"
    )
    n = {tag: len(bundle["runs"][tag]["curve"]) for tag in TAGS}
    print(f"wrote {args.out} — curve points {n}, t_max {bundle['t_max']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
