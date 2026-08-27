"""Fault-model audit: does our injector reproduce the measured GPU distribution?

This is the artifact that answers the circularity objection. Until now the
runtime's only evidence was that its detector caught the faults its own
injector produced, with both written by us and calibrated by us. A reviewer is
right to discount that.

Tung et al. (NVIDIA, DSN 2026, arXiv 2605.04213) published an outcome
distribution measured over 600 million corruptions from 3M+ simulator hours on
a production-class datacenter GPU. That is external ground truth. This script
draws events from our injector and compares the resulting distribution against
theirs, per path.

Two distributions are checked, because two different physics apply:

  * COMPUTE -- activations, values in flight. This is what Tung et al.
    measured (fault injection into hardware units, observing SM output), so the
    published shares apply directly and a mismatch is a defect.
  * MEMORY -- parameters and optimizer state, tensors at rest. Control-logic
    tile faults cannot reach these, so the published compute shares do NOT
    apply. Governed by the memory-array literature (MICRO'21). Reported for
    transparency, checked against the modelled shares, and explicitly NOT
    checked against the compute numbers.

Run: python bench/fault_model_audit.py [--events 20000] [--json out.json]
Exit code is non-zero if the compute path is outside tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np
import torch
from torch import nn

from orbital_runtime.inject.compute import ComputeInjector
from orbital_runtime.inject.gpu_model import (
    CLASS_BITFLIP,
    CLASS_NULLIFICATION,
    CLASS_SPECIAL,
    GPU_SINGLE_BIT_SHARE,
    PATH_MEMORY,
    WARP_STRIDES,
    expected_shares,
)
from orbital_runtime.inject.memory import MemoryInjector

# Absolute tolerance on each share. The published figures are quoted to two
# decimals over 25,000 SDC cases; 1.5 points is comfortably inside sampling
# noise at 20k draws while still catching a real drift.
TOLERANCE = 0.015


def _model():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 128))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    m(torch.randn(8, 128)).sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return m, opt


def audit_memory(n: int) -> dict:
    m, opt = _model()
    inj = MemoryInjector(m, opt)
    rng = np.random.default_rng(0)
    classes: Counter[str] = Counter()
    elements, nonfinite, bits = [], 0, Counter()
    for _ in range(n):
        ev = inj.inject_gpu_event(rng)
        if ev is None:
            continue
        classes[ev.fault_class] += 1
        elements.append(ev.n_elements)
        nonfinite += int(ev.became_nonfinite)
        for b in ev.bit_positions:
            bits[b] += 1
    total = sum(classes.values())
    return {
        "path": "memory",
        "events": total,
        "observed": {k: v / total for k, v in classes.items()},
        "modelled": expected_shares(PATH_MEMORY),
        "mean_elements": float(np.mean(elements)),
        "max_elements": int(max(elements)),
        "nonfinite_share": nonfinite / total,
        "bit_histogram": dict(sorted(bits.items())),
    }


def audit_compute(n: int) -> dict:
    m, _ = _model()
    inj = ComputeInjector(m).attach()
    rng = np.random.default_rng(0)
    x = torch.randn(8, 128)
    for _ in range(n):
        inj.arm(rng)
        m(x)
    hits = inj.drain_hits()
    classes: Counter[str] = Counter()
    elements, strides = [], Counter()
    for h in hits:
        classes[h.fault_class] += 1
        elements.append(h.n_elements)
        strides[h.stride] += 1
    total = sum(classes.values())
    single_elem = sum(1 for e in elements if e == 1)
    return {
        "path": "compute",
        "events": total,
        "observed": {k: v / total for k, v in classes.items()},
        "published": expected_shares(),
        "mean_elements": float(np.mean(elements)),
        "max_elements": int(max(elements)),
        "single_element_share": single_elem / total,
        "warp_stride_share": sum(strides[s] for s in WARP_STRIDES) / total,
        "strides": dict(sorted(strides.items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=20000)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    comp = audit_compute(args.events)
    mem = audit_memory(args.events)

    print("=" * 74)
    print("FAULT MODEL AUDIT -- injector vs Tung et al. 2026 (arXiv 2605.04213)")
    print("=" * 74)

    print(f"\n[COMPUTE PATH]  activations / values in flight   n={comp['events']}")
    print("  This is what the paper measured. Mismatch here is a defect.\n")
    print(f"  {'outcome':18s} {'ours':>9s} {'published':>11s} {'delta':>9s}   verdict")
    failed = []
    for k in (CLASS_NULLIFICATION, CLASS_BITFLIP, CLASS_SPECIAL):
        got = comp["observed"].get(k, 0.0)
        exp = comp["published"][k]
        d = got - exp
        ok = abs(d) <= TOLERANCE
        if not ok:
            failed.append(k)
        print(
            f"  {k:18s} {100*got:8.2f}% {100*exp:10.2f}% {100*d:+8.2f}pp   "
            f"{'PASS' if ok else 'FAIL'}"
        )
    print(f"\n  mean elements/event : {comp['mean_elements']:.1f}  (max {comp['max_elements']})")
    print(f"  single-element share: {100*comp['single_element_share']:.1f}%")
    print(f"  warp-aligned share  : {100*comp['warp_stride_share']:.1f}%"
          f"   strides {sorted(k for k in comp['strides'] if k > 1)}")

    print(f"\n[MEMORY PATH]  parameters / optimizer state   n={mem['events']}")
    print("  Tile faults cannot reach stored state; the compute shares do NOT")
    print("  apply here. Checked against the MICRO'21-derived model instead.\n")
    print(f"  {'outcome':18s} {'ours':>9s} {'modelled':>11s}")
    for k in (CLASS_NULLIFICATION, CLASS_BITFLIP, CLASS_SPECIAL):
        got = mem["observed"].get(k, 0.0)
        exp = mem["modelled"][k]
        print(f"  {k:18s} {100*got:8.2f}% {100*exp:10.2f}%")
    print(f"\n  mean elements/event : {mem['mean_elements']:.2f}  (max {mem['max_elements']})")
    print(f"  non-finite share    : {100*mem['nonfinite_share']:.2f}%"
          f"   (published NaN/Inf outcome share: 1.01%)")

    hist = mem["bit_histogram"]
    real_bits = {k: v for k, v in hist.items() if k >= 0}
    if real_bits:
        tot = sum(real_bits.values())
        low = sum(v for k, v in real_bits.items() if k < 16) / tot
        print(f"  bit positions < 16  : {100*low:.1f}%"
              f"   (Tung: flip probability decreases LSB -> MSB)")

    print("\n" + "=" * 74)
    if failed:
        print(f"RESULT: FAIL -- compute path outside {100*TOLERANCE:.1f}pp on {failed}")
    else:
        print("RESULT: PASS -- compute path reproduces the published distribution")
    print("=" * 74)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"compute": comp, "memory": mem}, fh, indent=2, default=str)
        print(f"wrote {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
