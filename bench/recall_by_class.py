"""Detector recall, split by measured GPU fault class.

The number Phase 1 exists to produce. Coverage says what the detectors CAN see;
this says what they DO catch, against the fault distribution Tung et al. actually
measured rather than the one the injector used to assume.

Method, one trial:

  1. build model + optimizer + full detector stack, take a clean step so Adam
     materialises its state, snapshot all baselines
  2. inject exactly ONE event of a chosen class into a chosen path
  3. run one step in the REAL loop order:
        forward -> backward -> check_now() -> optimizer.step() -> refresh()
  4. record which tier, if any, fired

Reported per class, because a single blended recall number hides the thing that
matters: a detector can look strong overall while being blind to the class that
dominates real faults.

Severity note. The integrity tier deliberately absorbs corruption below a
severity threshold instead of escalating (see detect/integrity.py). Those are
NOT misses and are counted separately as `absorbed`. Two recalls are reported:

  * detection recall  -- the fault was SEEN (escalated or absorbed)
  * escalation recall -- the fault was acted on

Honest accounting requires both. A run that absorbs a 5e-7 perturbation in an
Adam moment has not failed to detect it.

Run: python bench/recall_by_class.py [--trials 200] [--device cuda] [--json out]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from orbital_runtime.detect import Detector, GuardTier, IntegrityTier
from orbital_runtime.detect.abft import AbftTier
from orbital_runtime.inject.compute import ComputeInjector
from orbital_runtime.inject.gpu_model import (
    CLASS_BITFLIP,
    CLASS_NULLIFICATION,
    CLASS_SPECIAL,
    PATH_COMPUTE,
    PATH_MEMORY,
    FaultClass,
    sample_fault_class,
)
from orbital_runtime.inject.memory import MemoryInjector
from orbital_runtime.workload import get_workload, resolve_device

CLASSES = (CLASS_NULLIFICATION, CLASS_BITFLIP, CLASS_SPECIAL)


def _fresh(device, seed: int, compute_injector: bool = False, **kw):
    """Build workload + detector stack in PRODUCTION hook order.

    Hook order is load-bearing and easy to get backwards. `register_forward_hook`
    fires hooks in registration order, so if the ABFT tier attaches BEFORE the
    compute injector, ABFT verifies the clean output and the corruption lands
    afterwards -- ABFT is then structurally incapable of seeing an activation
    fault, and measured compute recall collapses to ~1%.

    `run.py` builds the RadiationEnvironment (which owns the compute injector)
    before the Detector, so in production the injector hooks FIRST. This mirrors
    that. Getting it wrong here does not produce an error, it produces a
    plausible and completely wrong number.
    """
    torch.manual_seed(seed)
    wl = get_workload("nanogpt", device=device, seed=seed, **kw)
    wl.loss_for_step(0).backward()
    wl.optimizer.step()
    wl.optimizer.zero_grad(set_to_none=True)

    ci = None
    if compute_injector:
        ci = ComputeInjector(wl.model).attach()  # FIRST, as in run.py

    abft = AbftTier(
        wl.model,
        rng=np.random.default_rng(seed),
        base_sample_rate=1.0,
        saa_sample_rate=1.0,
        adaptive=False,
    ).attach()
    abft.refresh_checksums()
    integrity = IntegrityTier(wl.model, optimizer=wl.optimizer)
    integrity.refresh()
    det = Detector(guards=GuardTier(), abft=abft, integrity=integrity)
    return wl, det, ci


def _draw_of_class(rng, target_class: str, numel: int, path: str) -> FaultClass:
    """Rejection-sample the mechanism sampler until it yields `target_class`.

    Uses the real sampler rather than hand-built geometry, so the geometry a
    class gets here is the geometry it gets in production.
    """
    for _ in range(500):
        fc = sample_fault_class(rng, numel=numel, path=path)
        if fc.label == target_class:
            return fc
    raise RuntimeError(f"sampler never produced {target_class} on {path}")


def trial(device, seed: int, target_class: str, path: str, **kw) -> str:
    """Returns one of: escalated | absorbed | missed | masked."""
    wl, det, ci = _fresh(
        device, seed, compute_injector=(path == PATH_COMPUTE), **kw
    )
    rng = np.random.default_rng(seed + 9001)

    if path == PATH_MEMORY:
        inj = MemoryInjector(wl.model, wl.optimizer)
        ev = None
        for _ in range(200):
            ev = inj.inject_gpu_event(rng)
            if ev is not None and ev.fault_class == target_class:
                break
            ev = None
        if ev is None:
            return "masked"
    else:
        assert ci is not None
        probe = max(p.numel() for p in wl.model.parameters())
        ci.force_fault_class = _draw_of_class(rng, target_class, probe, PATH_COMPUTE)
        ci.arm(rng)

    det.abft.arm()
    loss = wl.loss_for_step(1)
    loss.backward()

    # REAL loop order: integrity verifies before the optimizer moves anything.
    iv = det.integrity.check_now(1)
    wl.optimizer.step()
    v = det.observe(
        step=1,
        loss=float(loss.detach()),
        grad_norm=0.0,
        model=wl.model,
    )

    if v.triggered or iv.triggered:
        return "escalated"
    if det.integrity.stats.mismatches > 0:
        return "absorbed"
    return "missed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default=None)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    args = ap.parse_args()

    device = resolve_device(args.device)
    kw = dict(n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head)

    results: dict = defaultdict(lambda: defaultdict(int))
    for path in (PATH_MEMORY, PATH_COMPUTE):
        for cls in CLASSES:
            for i in range(args.trials):
                try:
                    out = trial(device, 1000 + i, cls, path, **kw)
                except Exception:
                    out = "masked"
                results[f"{path}/{cls}"][out] += 1

    print("=" * 78)
    print(f"DETECTOR RECALL BY FAULT CLASS   device={device}  trials={args.trials}")
    print("=" * 78)
    print(f"\n{'path / class':28s} {'detect':>8s} {'escalate':>9s} "
          f"{'absorbed':>9s} {'missed':>8s}")
    summary = {}
    for key, counts in results.items():
        tot = sum(counts.values()) - counts.get("masked", 0)
        if tot <= 0:
            continue
        esc = counts.get("escalated", 0)
        abso = counts.get("absorbed", 0)
        miss = counts.get("missed", 0)
        det_recall = (esc + abso) / tot
        esc_recall = esc / tot
        summary[key] = {
            "n": tot,
            "detection_recall": det_recall,
            "escalation_recall": esc_recall,
            "absorbed": abso,
            "missed": miss,
        }
        print(f"{key:28s} {100*det_recall:7.1f}% {100*esc_recall:8.1f}% "
              f"{abso:9d} {miss:8d}")

    # Blended, weighted by the published outcome shares -- the headline number.
    from orbital_runtime.inject.gpu_model import expected_shares

    for path in (PATH_MEMORY, PATH_COMPUTE):
        shares = expected_shares(path)
        num = den = 0.0
        for cls, w in shares.items():
            k = f"{path}/{cls}"
            if k in summary:
                num += w * summary[k]["detection_recall"]
                den += w
        if den:
            print(f"\n{path:>10s} share-weighted DETECTION recall: {100*num/den:.1f}%")

    print("\nFloor for a product: 70% (NVIDIA EUD's reported production recall).")
    print("Target: 90%.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
