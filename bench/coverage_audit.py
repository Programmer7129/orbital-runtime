"""Coverage audit: what fraction of the fault space can ABFT possibly see?

Phase 1, gate 2. Answers two questions the runtime has never answered:

1. COVERAGE -- of all the state a memory SEU can strike, and of all the
   compute a logic fault can corrupt, how much sits inside an op the ABFT
   tier checks? ABFT snapshots `s = W.sum(dim=0)` for `nn.Linear` only, so
   everything else is structurally invisible to it.

2. SENSITIVITY -- inside a checked op, how large must a corruption be before
   the V-ABFT tolerance calls it a fault? Anything smaller passes silently.

Coverage is reported three ways, because the number differs a lot by weighting
and only one of them is the honest one for a position-adaptive design:

  * structural   -- share of strikeable bits (or FLOPs) inside a checked op,
                    ignoring sampling. The ceiling.
  * sampled      -- structural x sample rate, at a fixed orbital position.
  * fault-weighted -- structural x sample rate, integrated over the orbit and
                    weighted by where upsets actually ARRIVE. This is the
                    metric that matters: the tier deliberately spends its
                    budget inside the SAA where ~90% of upsets land.

Run: python bench/coverage_audit.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math

import torch
from torch import nn

from orbital_runtime.detect.abft import (
    DEFAULT_BASE_SAMPLE_RATE,
    DEFAULT_SAA_SAMPLE_RATE,
    DEFAULT_SAFETY_FACTOR,
    _tolerance,
)
from orbital_runtime.inject.memory import bits_of
from orbital_runtime.workload import get_workload


# --------------------------------------------------------------------------- #
# 1. Structural coverage over strikeable STATE (the memory-SEU channel)
# --------------------------------------------------------------------------- #

def state_coverage(workload) -> dict:
    """Split every strikeable resident bit into checked vs unchecked.

    Mirrors `MemoryInjector.targets()`: named parameters plus optimizer state
    tensors. ABFT's trusted checksum is taken over `nn.Linear.weight` only, so
    that is the sole checked category.
    """
    model, optimizer = workload.model, workload.optimizer

    linear_weight_ids = {
        id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)
    }

    buckets: dict[str, int] = {
        "linear_weight": 0,      # checked by the ABFT weight checksum
        "linear_bias": 0,        # subtracted out before checking -> unchecked
        "other_param": 0,        # embeddings, layernorm, ...
        "optimizer_state": 0,    # Adam exp_avg / exp_avg_sq
    }
    detail: dict[str, int] = {}

    for name, p in model.named_parameters():
        b = bits_of(p)
        detail[name] = b
        if id(p) in linear_weight_ids:
            buckets["linear_weight"] += b
        elif name.endswith(".bias") and any(
            id(p) is id(m.bias)
            for m in model.modules()
            if isinstance(m, nn.Linear) and m.bias is not None
        ):
            buckets["linear_bias"] += b
        else:
            buckets["other_param"] += b

    for group in optimizer.param_groups:
        for p in group["params"]:
            for key, tensor in optimizer.state.get(p, {}).items():
                if torch.is_tensor(tensor) and tensor.dtype.is_floating_point:
                    buckets["optimizer_state"] += bits_of(tensor)

    total = sum(buckets.values())
    checked = buckets["linear_weight"]
    return {
        "buckets_bits": buckets,
        "total_bits": total,
        "checked_bits": checked,
        "structural_coverage": (checked / total) if total else 0.0,
        "largest_params": dict(
            sorted(detail.items(), key=lambda kv: -kv[1])[:8]
        ),
    }


# --------------------------------------------------------------------------- #
# 2. Structural coverage over FLOPs (the compute/logic-fault channel)
# --------------------------------------------------------------------------- #

def flop_coverage(workload, n_steps: int = 2) -> dict:
    """Count forward FLOPs inside nn.Linear vs everything else.

    Uses torch's FlopCounterMode where available so the denominator includes
    attention SDPA, not just the linear layers. Falls back to a Linear-only
    count with `covered=None` if the counter is unavailable.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return {"available": False}

    model = workload.model
    linear_names = {
        name for name, m in model.named_modules() if isinstance(m, nn.Linear)
    }

    counter = FlopCounterMode(display=False, depth=None)
    with counter:
        loss = workload.loss_for_step(0)
        loss.backward()
    model.zero_grad(set_to_none=True)

    per_module = counter.get_flop_counts()
    global_counts = per_module.get("Global", {})
    total = sum(global_counts.values())

    # Sum the leaf Linear modules. FlopCounterMode keys modules by their
    # qualified path; a Linear leaf contributes mm/addmm ops under its own key.
    linear_flops = 0
    for mod_path, ops in per_module.items():
        if mod_path == "Global":
            continue
        leaf = mod_path.split(".", 1)[-1] if "." in mod_path else mod_path
        if leaf in linear_names or mod_path in linear_names:
            linear_flops += sum(ops.values())

    return {
        "available": True,
        "total_forward_backward_flops": total,
        "linear_flops": linear_flops,
        "structural_coverage": (linear_flops / total) if total else 0.0,
        "by_op": {str(k): v for k, v in global_counts.items()},
    }


# --------------------------------------------------------------------------- #
# 3. Fault-weighted coverage over the orbit
# --------------------------------------------------------------------------- #

def fault_weighted_coverage(
    structural: float,
    *,
    saa_share: float,
    base_rate: float = DEFAULT_BASE_SAMPLE_RATE,
    saa_rate: float = DEFAULT_SAA_SAMPLE_RATE,
) -> dict:
    """Weight sampling by where upsets ARRIVE, not by wall-clock time.

    `saa_share` is the fraction of upsets that land inside the SAA. The tier
    verifies at `saa_rate` there and `base_rate` outside, so the share of
    upsets that land in a verified op is the upset-weighted mean.
    """
    weighted_sample = saa_share * saa_rate + (1.0 - saa_share) * base_rate
    time_weighted = 0.1 * saa_rate + 0.9 * base_rate  # ~10% of orbit in SAA
    return {
        "saa_share_of_upsets": saa_share,
        "upset_weighted_sample_rate": weighted_sample,
        "time_weighted_sample_rate": time_weighted,
        "fault_weighted_coverage": structural * weighted_sample,
        "quiescent_only_coverage": structural * base_rate,
        "saa_only_coverage": structural * saa_rate,
    }


# --------------------------------------------------------------------------- #
# 4. Sensitivity: the smallest corruption V-ABFT can resolve
# --------------------------------------------------------------------------- #

def sensitivity(workload, *, safety: float = DEFAULT_SAFETY_FACTOR) -> dict:
    """Report the detection floor for each Linear, in relative terms.

    The tier flags a row when `|residual| > tol`, with
    `tol = scale * _tolerance(dtype, k, 1.0, safety)` and `scale` the per-row
    L1 magnitude of the summed terms. So a corruption is invisible unless it
    perturbs the row sum by more than that fraction of the row's L1 norm.

    Reported as `min_relative_delta`: the smallest detectable absolute change
    to the row sum, divided by the row's L1 scale. Multiply by a row's L1 norm
    to get the absolute floor in units of the activations.
    """
    model = workload.model
    rows = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        k = m.in_features
        dtype = m.weight.dtype
        coeff = _tolerance(dtype, k, 1.0, safety)
        rows.append(
            {
                "module": name,
                "in_features": k,
                "out_features": m.out_features,
                "dtype": str(dtype).replace("torch.", ""),
                "min_relative_delta": coeff,
            }
        )
    if not rows:
        return {"layers": []}
    coeffs = [r["min_relative_delta"] for r in rows]
    return {
        "safety_factor": safety,
        "layers": rows,
        "best_case": min(coeffs),
        "worst_case": max(coeffs),
        "median": sorted(coeffs)[len(coeffs) // 2],
    }


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--saa-share", type=float, default=0.9)
    args = ap.parse_args()

    torch.manual_seed(0)
    wl = get_workload("nanogpt")

    # One step so Adam materialises its state tensors; before this the
    # optimizer state is empty and the coverage denominator is wrong.
    loss = wl.loss_for_step(0)
    loss.backward()
    wl.optimizer.step()
    wl.optimizer.zero_grad(set_to_none=True)

    state = state_coverage(wl)
    # What the integrity tier actually tracks, measured rather than assumed.
    from orbital_runtime.detect.integrity import IntegrityTier
    itier = IntegrityTier(wl.model, optimizer=wl.optimizer)
    itier.refresh()
    integrity_bits = itier.stats.bits_covered
    integrity_cov = integrity_bits / state["total_bits"] if state["total_bits"] else 0.0
    flops = flop_coverage(wl)
    sens = sensitivity(wl)
    fw_state = fault_weighted_coverage(
        state["structural_coverage"], saa_share=args.saa_share
    )
    fw_flops = (
        fault_weighted_coverage(flops["structural_coverage"], saa_share=args.saa_share)
        if flops.get("available")
        else None
    )

    report = {
        "state_channel": state,
        "flop_channel": flops,
        "sensitivity": sens,
        "fault_weighted_state": fw_state,
        "fault_weighted_flops": fw_flops,
        "integrity_structural_coverage": integrity_cov,
        "integrity_bits": integrity_bits,
    }

    print("=" * 72)
    print("COVERAGE AUDIT -- what ABFT can structurally see")
    print("=" * 72)

    print("\n[1] STRIKEABLE STATE (memory SEU channel)")
    tot = state["total_bits"]
    for k, v in state["buckets_bits"].items():
        mark = "CHECKED  " if k == "linear_weight" else "unchecked"
        print(f"    {mark}  {k:18s} {v/8/1e6:10.2f} MB   {100*v/tot:5.1f}%")
    print(f"    -> ABFT structural coverage      : {100*state['structural_coverage']:.1f}%")
    print(f"    -> INTEGRITY structural coverage : {100*integrity_cov:.1f}%"
          f"   ({integrity_bits/8/1e6:.2f} MB, {itier.stats.tensors_tracked} tensors, unsampled)")

    if flops.get("available"):
        print("\n[2] FORWARD+BACKWARD FLOPs (compute/logic fault channel)")
        print(f"    total          {flops['total_forward_backward_flops']/1e9:10.2f} GFLOP")
        print(f"    in nn.Linear   {flops['linear_flops']/1e9:10.2f} GFLOP")
        print(f"    -> structural coverage: {100*flops['structural_coverage']:.1f}%")

    print("\n[3] SAMPLING (base=%.2f outside SAA, %.2f inside)"
          % (DEFAULT_BASE_SAMPLE_RATE, DEFAULT_SAA_SAMPLE_RATE))
    print(f"    upset-weighted sample rate: {fw_state['upset_weighted_sample_rate']:.3f}"
          f"   (assuming {100*args.saa_share:.0f}% of upsets land in SAA)")
    print(f"    time-weighted  sample rate: {fw_state['time_weighted_sample_rate']:.3f}")

    combined = (
        integrity_cov  # exact, every step, no sampling
        + (1.0 - integrity_cov) * state["structural_coverage"]
        * fw_state["upset_weighted_sample_rate"]
    )
    print("\n[4] EFFECTIVE COVERAGE -- state channel")
    print(f"    COMBINED (abft+integrity): {100*combined:.1f}%   <- THE NUMBER")
    print(f"    abft alone     : {100*fw_state['fault_weighted_coverage']:.1f}%")
    print(f"    inside SAA only: {100*fw_state['saa_only_coverage']:.1f}%")
    print(f"    quiescent only : {100*fw_state['quiescent_only_coverage']:.1f}%")
    if fw_flops:
        print("\n    EFFECTIVE COVERAGE -- FLOP channel")
        print(f"    fault-weighted : {100*fw_flops['fault_weighted_coverage']:.1f}%")

    print("\n[5] SENSITIVITY -- smallest resolvable perturbation")
    print(f"    (relative to each row's L1 magnitude; safety factor {sens['safety_factor']})")
    print(f"    best  {sens['best_case']:.3e}   median {sens['median']:.3e}"
          f"   worst {sens['worst_case']:.3e}")
    for r in sens["layers"][:6]:
        print(f"      {r['module']:28s} k={r['in_features']:5d} "
              f"{r['dtype']:8s} floor={r['min_relative_delta']:.3e}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
