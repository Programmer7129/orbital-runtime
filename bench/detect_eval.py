"""Detector precision / recall against known injected faults (PLAN.md M2).

Ground truth without a threshold
--------------------------------
Determinism (design rule 3) buys an exact oracle here. Because the flip
schedule is drawn from a stream independent of everything else, we can run
the SAME workload and SAME seed twice -- once at rate 0, once irradiated --
and the two loss curves are BIT-IDENTICAL until the first fault actually
propagates. So:

    corruption_step = first step where loss_irradiated != loss_clean

is exact. No tolerance, no judgement call, no "significantly different".
The detector is never consulted in building the oracle.

This also settles the question M1 raised about masking. A fault that ReLU
annihilates, or that lands in an Adam second moment and is normalised away,
leaves the loss curve untouched -- so it is not counted as something the
detector was obliged to catch. Recall is measured against faults that
PROPAGATE, which is the only kind that can hurt the model, rather than
against faults injected. Scoring masked faults as misses would understate
recall by punishing detectors for ignoring non-events.

Metrics
-------
Run-level, which is what the demo actually claims:
  * TP  corrupted AND detected            * FN  corrupted, never detected
  * FP  clean but detector fired          * TN  clean, detector silent
  * latency: steps from corruption_step to the first detection at/after it

Detectors are read-only in M2 (no recovery), so their presence cannot alter
the run being measured -- asserted in test_detect_eval.py.

Run:  python -m bench.detect_eval --seeds 12 --rate 5e-4
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from orbital_runtime.detect import Detector, GuardTier
from orbital_runtime.detect.abft import AbftTier
from orbital_runtime.detect.watcher import SimulatedXidSource, WatcherTier
from orbital_runtime.inject.injector import RadiationEnvironment
from orbital_runtime.inject.memory import MemoryInjector
from orbital_runtime.inject.xid import XidSimulator
from orbital_runtime.orbit.flux import FluxModel
from orbital_runtime.orbit.track import OrbitTrack
from orbital_runtime.train import TrainConfig, train
from orbital_runtime.workload import get_workload, resolve_device


@dataclass
class RunOutcome:
    seed: int
    corrupted: bool
    corruption_step: int | None
    detected: bool
    first_detection_step: int | None
    latency: int | None
    died: bool
    flips: int
    tiers: dict[str, int] = field(default_factory=dict)
    first_reason: str = ""


def first_divergence(a: list[float], b: list[float]) -> int | None:
    """First index where two deterministic loss curves differ at all.

    NaN-aware: `nan != nan` is True in IEEE-754, so a naive `!=` would call
    two identically-dead curves "divergent" at their first NaN. The clean
    oracle run never NaNs, so this cannot bite today -- but the function is
    the definition of ground truth, and it should not depend on that.
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if math.isnan(x) and math.isnan(y):
            continue
        if x != y:
            return i
    # One run died early: divergence is where the shorter one stopped.
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def build_detector(model, *, tiers: str, xid_sim: XidSimulator | None) -> Detector:
    guards = GuardTier() if "guards" in tiers else None
    abft = (
        AbftTier(model, base_sample_rate=0.1, saa_sample_rate=1.0, adaptive=True).attach()
        if "abft" in tiers
        else None
    )
    watcher = (
        WatcherTier(source=SimulatedXidSource(xid_sim))
        if "watcher" in tiers and xid_sim is not None
        else None
    )
    return Detector(guards=guards, abft=abft, watcher=watcher)


def evaluate_seed(
    *, seed: int, rate: float, steps: int, orbits: float, device, tiers: str, workload_kw: dict
) -> RunOutcome:
    # --- the oracle: same seed, no radiation ---
    clean_w = get_workload("nanogpt", seed=seed, device=device, **workload_kw)
    clean = train(clean_w, cfg=TrainConfig(steps=steps))

    # --- the run under test ---
    w = get_workload("nanogpt", seed=seed, device=device, **workload_kw)
    bits = MemoryInjector(w.model, w.optimizer).static_resident_bits()
    flux = FluxModel(
        bits_resident=bits, track=OrbitTrack(), base_rate_upsets_per_bit_day=rate
    )
    xid_sim = XidSimulator(ecc_on=False)
    env = RadiationEnvironment(
        w.model,
        w.optimizer,
        flux=flux,
        seed=seed,
        n_steps=steps,
        orbits=orbits,
        xid=xid_sim,
    )
    detector = build_detector(w.model, tiers=tiers, xid_sim=xid_sim)
    result = train(w, cfg=TrainConfig(steps=steps), env=env, detector=detector)

    corruption_step = first_divergence(clean.losses, result.losses)
    corrupted = corruption_step is not None

    detections = detector.history
    first_det = detections[0].step if detections else None

    latency = None
    if corrupted and detections:
        after = [d.step for d in detections if d.step >= corruption_step]
        if after:
            latency = after[0] - corruption_step

    return RunOutcome(
        seed=seed,
        corrupted=corrupted,
        corruption_step=corruption_step,
        detected=bool(detections),
        first_detection_step=first_det,
        latency=latency,
        died=result.died,
        flips=env.stats.flips,
        tiers=dict(detector.per_tier),
        first_reason=detections[0].reason if detections else "",
    )


def summarise(outcomes: list[RunOutcome], label: str) -> dict:
    tp = sum(1 for o in outcomes if o.corrupted and o.detected)
    fn = sum(1 for o in outcomes if o.corrupted and not o.detected)
    fp = sum(1 for o in outcomes if not o.corrupted and o.detected)
    tn = sum(1 for o in outcomes if not o.corrupted and not o.detected)

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    lat = [o.latency for o in outcomes if o.latency is not None]

    return {
        "label": label,
        "runs": len(outcomes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "median_latency_steps": statistics.median(lat) if lat else None,
        "max_latency_steps": max(lat) if lat else None,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.detect_eval")
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--rate", type=float, default=5e-4)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--orbits", type=float, default=2.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-layer", type=int, default=1)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--n-embd", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    workload_kw = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        batch_size=args.batch_size,
        block_size=args.block_size,
    )

    print("=" * 78)
    print(f"detector evaluation | device={device} seeds={args.seeds} rate={args.rate:.0e}")
    print("  ground truth: exact divergence from a seed-matched clean run")
    print("=" * 78)

    configs = ["guards", "guards+abft", "guards+abft+watcher"]
    summaries = []
    all_outcomes: dict[str, list[RunOutcome]] = {}

    for tiers in configs:
        outcomes = [
            evaluate_seed(
                seed=s,
                rate=args.rate,
                steps=args.steps,
                orbits=args.orbits,
                device=device,
                tiers=tiers,
                workload_kw=workload_kw,
            )
            for s in range(1, args.seeds + 1)
        ]
        all_outcomes[tiers] = outcomes
        summaries.append(summarise(outcomes, tiers))
        print(f"  {tiers} done")

    # --- clean-run false positives: the honest cost of the statistical tiers ---
    print("\n  measuring false positives on CLEAN runs (rate=0)...")
    clean_outcomes = [
        evaluate_seed(
            seed=s,
            rate=0.0,
            steps=args.steps,
            orbits=args.orbits,
            device=device,
            tiers="guards+abft",
            workload_kw=workload_kw,
        )
        for s in range(1, args.seeds + 1)
    ]
    clean_fp = sum(1 for o in clean_outcomes if o.detected)

    print("\n" + "=" * 78)
    print(f"{'tiers':26} {'prec':>6} {'recall':>7} {'TP':>3} {'FP':>3} {'FN':>3} {'lat(med)':>9}")
    print("-" * 78)
    for s in summaries:
        lat = "-" if s["median_latency_steps"] is None else f"{s['median_latency_steps']:.0f}"
        print(
            f"{s['label']:26} {s['precision']:6.2f} {s['recall']:7.2f} "
            f"{s['tp']:3d} {s['fp']:3d} {s['fn']:3d} {lat:>9}"
        )
    print("=" * 78)
    print(
        f"\nfalse positives on {len(clean_outcomes)} clean (unirradiated) runs: "
        f"{clean_fp}/{len(clean_outcomes)}"
    )

    corrupted = sum(1 for o in all_outcomes[configs[0]] if o.corrupted)
    died = sum(1 for o in all_outcomes[configs[0]] if o.died)
    print(f"irradiated runs corrupted: {corrupted}/{args.seeds} (died: {died})")

    best = all_outcomes["guards+abft+watcher"]
    by_reason: dict[str, int] = {}
    for o in best:
        if o.first_reason:
            by_reason[o.first_reason] = by_reason.get(o.first_reason, 0) + 1
    print(f"first-detection reason: {by_reason}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "rate": args.rate,
                    "seeds": args.seeds,
                    "steps": args.steps,
                    "device": str(device),
                    "summaries": summaries,
                    "clean_false_positives": clean_fp,
                    "clean_runs": len(clean_outcomes),
                    "outcomes": {k: [asdict(o) for o in v] for k, v in all_outcomes.items()},
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
