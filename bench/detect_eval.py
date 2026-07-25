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

import torch

from orbital_runtime.detect import Detector, GuardTier
from orbital_runtime.detect.abft import AbftTier
from orbital_runtime.detect.watcher import SimulatedXidSource, WatcherTier
from orbital_runtime.inject.injector import RadiationEnvironment
from orbital_runtime.inject.memory import MemoryInjector
from orbital_runtime.inject.sefi import SefiInjector
from orbital_runtime.inject.xid import XidSimulator
from orbital_runtime.orbit.flux import FluxModel
from orbital_runtime.orbit.track import OrbitTrack
from orbital_runtime.train import TrainConfig, train
from orbital_runtime.workload import get_workload, resolve_device


# --------------------------------------------------------------------------- #
# Clopper-Pearson exact binomial confidence intervals
# --------------------------------------------------------------------------- #
# Every ratio here (recall, false-positive rate, corrupted fraction) is a
# binomial proportion from a handful of runs. Reporting "6/6 = 1.00" without an
# interval overstates certainty: 6/6 is consistent with a true rate anywhere
# down to ~0.54 at 95%. Clopper-Pearson is the exact (conservative) interval,
# implemented here from the regularized incomplete beta so the bench has no
# scipy dependency (Numerical Recipes betacf; bisection for the inverse).


def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) <= EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(p: float, a: float, b: float) -> float:
    """Inverse of I_x(a, b) = p by bisection (monotone in x)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _betai(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided 100*(1-alpha)% CI for k successes in n Bernoulli trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else _beta_ppf(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else _beta_ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return (lo, hi)


def _fmt_ratio(k: int, n: int) -> str:
    """`k/n = 0.83 (95% CI 0.52-0.98)` -- the honest way to state a small ratio."""
    if n == 0:
        return f"{k}/{n} = n/a"
    lo, hi = clopper_pearson(k, n)
    return f"{k}/{n} = {k / n:.2f} (95% CI {lo:.2f}-{hi:.2f})"


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
        # SEFI OFF here: detect_eval measures MEMORY-fault detection recall
        # against the exact loss-divergence oracle. SEFIs are a separate crash
        # channel with their own recovery test; letting them fire would inject
        # process deaths that no detection tier is meant to "catch", muddying
        # the recall measurement. (SEFI is on by default in the product path.)
        sefi=SefiInjector(flux.track, p_per_transit=0.0),
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
    # A corrupted run is a TRUE POSITIVE only if the detector fired AT OR AFTER
    # the corruption step. `o.latency is not None` encodes exactly that (see
    # evaluate_seed: latency is the gap to the first detection at/after
    # corruption). A detection that PRECEDED the corruption catches nothing --
    # it is an early (spurious) fire, folded into FN for recall and surfaced
    # separately. This is the scoring bug the hostile review flagged (item 1):
    # `detected=bool(detections)` scored a pre-corruption blip as a hit.
    tp = sum(1 for o in outcomes if o.corrupted and o.latency is not None)
    fn = sum(1 for o in outcomes if o.corrupted and o.latency is None)
    fp = sum(1 for o in outcomes if not o.corrupted and o.detected)
    tn = sum(1 for o in outcomes if not o.corrupted and not o.detected)
    # Corrupted runs whose only detection came BEFORE the corruption.
    early = sum(
        1 for o in outcomes if o.corrupted and o.detected and o.latency is None
    )

    # Precision here is VACUOUS on a pure irradiated cohort (fp = tn = 0 by
    # construction: there are no clean runs to raise a false positive), so it is
    # always 1.0 whenever tp > 0 and says nothing. Kept for the mixed-cohort
    # unit test; the headline precision proxy is the clean-run FP rate reported
    # by main(). Recall, by contrast, is a real measurement.
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    recall_lo, recall_hi = clopper_pearson(tp, tp + fn) if (tp + fn) else (float("nan"), float("nan"))
    lat = [o.latency for o in outcomes if o.latency is not None]

    return {
        "label": label,
        "runs": len(outcomes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "early_detections": early,
        "precision": precision,
        "precision_vacuous": (fp + tn) == 0,
        "recall": recall,
        "recall_ci95": [recall_lo, recall_hi],
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
    clean_n = len(clean_outcomes)
    clean_fp_lo, clean_fp_hi = clopper_pearson(clean_fp, clean_n) if clean_n else (float("nan"), float("nan"))

    print("\n" + "=" * 78)
    print(f"{'tiers':26} {'recall (95% CI)':>22} {'TP':>3} {'FN':>3} {'early':>5} {'lat(med)':>9} {'lat(max)':>9}")
    print("-" * 78)
    for s in summaries:
        lat = "-" if s["median_latency_steps"] is None else f"{s['median_latency_steps']:.0f}"
        latmax = "-" if s["max_latency_steps"] is None else f"{s['max_latency_steps']:.0f}"
        rc = _fmt_ratio(s["tp"], s["tp"] + s["fn"]).split(" = ")[1]  # "0.83 (95% CI ...)"
        print(
            f"{s['label']:26} {rc:>22} {s['tp']:3d} {s['fn']:3d} "
            f"{s['early_detections']:5d} {lat:>9} {latmax:>9}"
        )
    print("=" * 78)
    print(
        "\nprecision proxy = false-positive rate on CLEAN (unirradiated) runs "
        "(the irradiated-cohort precision is vacuous: fp=tn=0 by construction):"
    )
    print(f"  clean-run false positives: {_fmt_ratio(clean_fp, clean_n)}")

    corrupted = sum(1 for o in all_outcomes[configs[0]] if o.corrupted)
    died = sum(1 for o in all_outcomes[configs[0]] if o.died)
    print(f"irradiated runs corrupted: {_fmt_ratio(corrupted, args.seeds)} (died: {died})")

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
                    # Provenance (item 19): model size + config + torch version
                    # live inside the artifact, so a committed JSON is
                    # self-describing and cannot be misread as a different scale.
                    "config": {
                        "n_layer": args.n_layer,
                        "n_head": args.n_head,
                        "n_embd": args.n_embd,
                        "batch_size": args.batch_size,
                        "block_size": args.block_size,
                        "orbits": args.orbits,
                        "torch_version": torch.__version__,
                    },
                    "scoring_note": (
                        "TP requires first detection at or after the corruption "
                        "step (item 1 fix). Precision proxy is the clean-run FP "
                        "rate below; irradiated-cohort precision is vacuous "
                        "(fp=tn=0). All ratios carry Clopper-Pearson 95% CIs."
                    ),
                    "summaries": summaries,
                    "clean_false_positives": clean_fp,
                    "clean_runs": clean_n,
                    "clean_fp_rate": (clean_fp / clean_n) if clean_n else None,
                    "clean_fp_ci95": [clean_fp_lo, clean_fp_hi],
                    "corrupted": corrupted,
                    "corrupted_ci95": list(clopper_pearson(corrupted, args.seeds)),
                    "outcomes": {k: [asdict(o) for o in v] for k, v in all_outcomes.items()},
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
