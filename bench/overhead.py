"""Per-tier overhead measurement (PLAN.md M2 + design rule 4).

"Report measured overhead per tier; target <10% total with ABFT sampling on.
If we can't hit it, report what we hit -- no fudging."

Method
------
Each configuration trains the SAME workload from the SAME seed for the same
number of steps, with radiation OFF. Radiation is off deliberately: a run
that dies early would report a fraudulently cheap "overhead", and a run
whose weights explode changes the arithmetic the hardware is doing. We are
measuring the cost of LOOKING, not the cost of being hit.

Two hard-won methodology points -- the naive version of this benchmark
reported NEGATIVE overhead for tier 1, which is impossible (watching two
scalars cannot make training faster) and was pure measurement noise:

  * **Round-robin interleaving.** Timing all of config A and then all of
    config B lets slow drift (thermal throttling, another process waking up)
    masquerade as a difference between configs. Cycling A,B,C,A,B,C spreads
    any drift across all of them.

  * **An A/A control.** The baseline is timed TWICE under different names.
    The two are identical code, so their apparent "overhead" is pure noise,
    and it is the resolution limit of the whole experiment. Any effect
    smaller than the control is NOT MEASURABLE on this machine, and is
    reported as such rather than as a number. Without this control there is
    no way to tell a real 1% from a noisy 0%.

Other hygiene: a warmup phase is discarded (lazy init, allocator warmup, MPS
shader compilation); medians rather than means, because laptop noise is
one-sided; MPS/CUDA are synchronised before reading the clock, or the queue
drains after the timer stops and every number is fiction.

Run:  python -m bench.overhead --steps 200 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from orbital_runtime.detect import Detector, GuardTier
from orbital_runtime.detect.abft import AbftTier
from orbital_runtime.workload import get_workload, resolve_device


def _sync(device: torch.device) -> None:
    """Make the clock mean something on an async backend."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class Timing:
    name: str
    median_step_ms: float
    mean_step_ms: float
    p90_step_ms: float
    total_s: float
    overhead_pct: float = 0.0
    notes: str = ""


def time_config(
    *,
    name: str,
    device: torch.device,
    steps: int,
    warmup: int,
    seed: int,
    make_detector,
    workload_kw: dict,
) -> Timing:
    """Time one configuration, warmup discarded."""
    workload = get_workload("nanogpt", seed=seed, device=device, **workload_kw)
    model, optimizer = workload.model, workload.optimizer
    model.train()

    detector = make_detector(workload)
    step_times: list[float] = []

    total_start = time.perf_counter()
    for step in range(warmup + steps):
        if step == warmup:
            step_times.clear()  # discard warmup

        t0 = time.perf_counter()

        if detector is not None:
            # Sweep the whole orbit so adaptive sampling is exercised at
            # both its rates, in proportion -- timing only the SAA would
            # report the worst case as if it were the average.
            in_saa = (step % 95) < 10
            detector.before_step(t_sim=float(step), in_saa=in_saa)

        loss = workload.loss_for_step(step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()

        if detector is not None:
            if detector.abft is not None:
                detector.abft.refresh_checksums()
            detector.observe(
                step=step, loss=float(loss.item()), grad_norm=grad_norm, model=model
            )

        _sync(device)
        step_times.append((time.perf_counter() - t0) * 1000.0)

    total = time.perf_counter() - total_start

    if detector is not None and detector.abft is not None:
        detector.abft.detach()

    ordered = sorted(step_times)
    return Timing(
        name=name,
        median_step_ms=statistics.median(step_times),
        mean_step_ms=statistics.fmean(step_times),
        p90_step_ms=ordered[int(0.9 * len(ordered))],
        total_s=total,
    )


BASELINE = "baseline (unprotected)"
CONTROL = "A/A control (baseline again)"
ADAPTIVE = "tier1+2: guards + ABFT (adaptive)"


def build_configs():
    """The tiers, cumulative, so each row isolates one addition."""

    def none(_w):
        return None

    def guards_only(_w):
        return Detector(guards=GuardTier())

    def guards_abft_adaptive(w):
        return Detector(
            guards=GuardTier(),
            abft=AbftTier(
                w.model, base_sample_rate=0.1, saa_sample_rate=1.0, adaptive=True
            ).attach(),
        )

    def guards_abft_full(w):
        return Detector(
            guards=GuardTier(),
            abft=AbftTier(w.model, base_sample_rate=1.0, adaptive=False).attach(),
        )

    return [
        (BASELINE, none),
        # Identical to BASELINE. Its apparent overhead is the noise floor.
        (CONTROL, none),
        ("tier1: guards", guards_only),
        (ADAPTIVE, guards_abft_adaptive),
        ("tier1+2: guards + ABFT (100% sampling)", guards_abft_full),
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.overhead")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    workload_kw = dict(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        batch_size=args.batch_size,
        block_size=args.block_size,
    )

    print("=" * 78)
    print(f"overhead benchmark | device={device} steps={args.steps} repeats={args.repeats}")
    print(f"  warmup={args.warmup} discarded | radiation OFF (measuring the cost of looking)")
    print("=" * 78)

    configs = build_configs()
    samples: dict[str, list[Timing]] = {name: [] for name, _ in configs}

    # Round-robin: drift hits every configuration equally.
    for rep in range(args.repeats):
        for name, make in configs:
            samples[name].append(
                time_config(
                    name=name,
                    device=device,
                    steps=args.steps,
                    warmup=args.warmup,
                    seed=args.seed,
                    make_detector=make,
                    workload_kw=workload_kw,
                )
            )
        print(f"  repeat {rep + 1}/{args.repeats} done")

    results: list[Timing] = []
    for name, _ in configs:
        runs = samples[name]
        results.append(
            Timing(
                name=name,
                median_step_ms=statistics.median(r.median_step_ms for r in runs),
                mean_step_ms=statistics.median(r.mean_step_ms for r in runs),
                p90_step_ms=statistics.median(r.p90_step_ms for r in runs),
                total_s=statistics.median(r.total_s for r in runs),
            )
        )

    by_name = {r.name: r for r in results}
    base = by_name[BASELINE].median_step_ms
    for r in results:
        r.overhead_pct = 100.0 * (r.median_step_ms - base) / base

    # The control is identical code to the baseline, so whatever it
    # "measures" is the experiment's resolution limit.
    noise_floor = abs(by_name[CONTROL].overhead_pct)

    print("\n" + "=" * 78)
    print(f"{'configuration':42} {'ms/step':>9} {'overhead':>10}")
    print("-" * 78)
    for r in results:
        if r.name == BASELINE:
            pct = "--"
        elif abs(r.overhead_pct) <= noise_floor:
            pct = "<noise"
            r.notes = f"below the {noise_floor:.1f}% noise floor; not resolvable"
        else:
            pct = f"{r.overhead_pct:+.1f}%"
        print(f"{r.name:42} {r.median_step_ms:9.3f} {pct:>10}")
    print("-" * 78)
    print(
        f"{'noise floor (A/A control)':42} {'':9} {noise_floor:9.1f}%"
        "   <- resolution limit"
    )
    print("=" * 78)

    target = by_name[ADAPTIVE]
    print("\nPLAN.md target: <10% total with ABFT sampling on.")
    if target.overhead_pct >= 10.0:
        print(f"  MISSES: {target.overhead_pct:+.1f}% measured.")
        print("  Reported as measured -- design rule 4 (overhead honesty).")
    elif target.overhead_pct > noise_floor:
        print(f"  MEETS: {target.overhead_pct:+.1f}% measured (noise floor {noise_floor:.1f}%).")
    else:
        print(
            f"  MEETS: the cost of the shipped configuration is below this "
            f"machine's {noise_floor:.1f}% noise floor,"
        )
        print(
            "  i.e. too small to measure here -- an upper bound, not a claim of zero."
        )
        print("  A real per-step cost needs a quieter machine (M4 cloud GPU).")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "device": str(device),
                    "steps": args.steps,
                    "repeats": args.repeats,
                    "results": [asdict(r) for r in results],
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
