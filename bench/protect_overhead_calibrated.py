"""Protected-run WALL-CLOCK overhead at CALIBRATED rates -- CONTROLLED rerun.

Design rule 4 (overhead honesty) done properly, matching the methodology of
`bench/overhead.py`:

  * **>=5 round-robin-interleaved repeats.** Timing all of one condition and
    then all of the next lets slow drift (thermal throttling, another process
    waking up) masquerade as a difference between conditions. We cycle
    baseline, control, 1e-9, 1e-8, 1e-7, baseline, control, ... so any drift
    is spread across every condition instead of loading onto one.
  * **An A/A control.** The clean baseline is timed TWICE under two names. The
    two are identical code, so their apparent "overhead" is pure timing noise
    and IS the resolution limit of the experiment. Any protection overhead
    smaller than the control is not measurable on this machine and is reported
    as "<noise", never as a number.
  * **A discarded warmup round** (lazy init, allocator warmup, MPS shader /
    cuDNN autotune) and **medians, not means** (machine noise is one-sided),
    with the accelerator **synchronised** before the clock is read.

Why this is harder than `bench/overhead.py`: that benchmark measures the cost
of LOOKING with radiation OFF, so every configuration does identical
arithmetic. Here radiation is ON at each calibrated flight-band rate, so a
protected run also pays for checkpoint I/O and any rollback/replay -- the full
wall-clock cost the M2 detection-only table deliberately excludes. The final
val loss is recorded alongside each row because a band-top survivor can finish
DEGRADED (sub-detection-floor accumulation, STATUS M3); "survived" without the
loss hides that.

Status: the committed `bench/results/protect_overhead_l4.json` is a **2-repeat,
non-interleaved, no-control, single-seed INDICATIVE** measurement. This script
is its controlled replacement, ready for the next GPU session. Nothing here is
CUDA-specific -- `--device cpu --smoke` exercises the whole path on a laptop.

Run (GPU session):
  python -m bench.protect_overhead_calibrated --device cuda \
      --n-layer 12 --n-head 12 --n-embd 768 --block-size 256 \
      --steps 150 --orbits 4 --repeats 5 \
      --json bench/results/protect_overhead_calibrated_l4.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from orbital_runtime.ckpt.policy import CheckpointPolicy
from orbital_runtime.ckpt.recover import RecoveryOrchestrator
from orbital_runtime.ckpt.saver import CheckpointSaver
from orbital_runtime.detect import Detector, GuardTier
from orbital_runtime.detect.abft import AbftTier
from orbital_runtime.detect.watcher import SimulatedXidSource, WatcherTier
from orbital_runtime.inject.injector import RadiationEnvironment
from orbital_runtime.inject.memory import MemoryInjector
from orbital_runtime.inject.xid import XidSimulator
from orbital_runtime.orbit.flux import FluxModel
from orbital_runtime.orbit.track import OrbitTrack
from orbital_runtime.rng import STREAM_ABFT, stream
from orbital_runtime.train import TrainConfig, train
from orbital_runtime.workload import get_workload, resolve_device

# The calibrated flight-band rates (upsets/bit-day). These are the sacred
# calibration (PLAN.md design rule 2), not elevated demo rates.
CALIBRATED_RATES = (1e-9, 1e-8, 1e-7)

BASELINE = "baseline (clean, protect off)"
CONTROL = "A/A control (baseline again)"


def _sync(device: torch.device) -> None:
    """Make the clock mean something on an async backend."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _time_baseline(device: torch.device, model_kw: dict, seed: int, steps: int) -> float:
    """Wall time of a clean run: no injector, no detector, no checkpoints."""
    w = get_workload("nanogpt", seed=seed, device=device, **model_kw)
    t0 = time.perf_counter()
    train(w, cfg=TrainConfig(steps=steps))
    _sync(device)
    return time.perf_counter() - t0


def _time_protected(
    device: torch.device,
    model_kw: dict,
    seed: int,
    steps: int,
    orbits: float,
    rate: float,
    ckpt_dir: Path,
):
    """Wall time + recovery outcome of a protected run under radiation at `rate`.

    Returns (wall_s, train_result, recovery_stats, env).
    """
    w = get_workload("nanogpt", seed=seed, device=device, **model_kw)
    bits = MemoryInjector(w.model, w.optimizer).static_resident_bits()
    flux = FluxModel(
        bits_resident=bits, track=OrbitTrack(), base_rate_upsets_per_bit_day=rate
    )
    xid = XidSimulator(ecc_on=False)
    env = RadiationEnvironment(
        w.model, w.optimizer, flux=flux, seed=seed, n_steps=steps, orbits=orbits, xid=xid
    )
    abft = AbftTier(
        w.model, flux=flux, base_sample_rate=0.1, saa_sample_rate=1.0,
        adaptive=True, rng=stream(seed, STREAM_ABFT),
    ).attach()
    det = Detector(
        guards=GuardTier(), abft=abft,
        watcher=WatcherTier(source=SimulatedXidSource(env.xid)),
    )
    saver = CheckpointSaver(w.model, w.optimizer, directory=ckpt_dir, use_async=True)
    rec = RecoveryOrchestrator(
        saver=saver,
        policy=CheckpointPolicy(
            track=OrbitTrack(), base_interval=50, saa_interval=10, adaptive=True
        ),
        env=env,
        detector=det,
    )
    t0 = time.perf_counter()
    r = train(w, cfg=TrainConfig(steps=steps), env=env, detector=det, recovery=rec)
    _sync(device)
    wall = time.perf_counter() - t0
    return wall, r, rec.stats_dict(), env


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.protect_overhead_calibrated")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--orbits", type=float, default=4.0)
    p.add_argument("--repeats", type=int, default=5, help="timed rounds (>=5 for design rule 4)")
    p.add_argument("--warmup-rounds", type=int, default=1, help="untimed rounds discarded up front")
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-embd", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--json", type=Path, default=Path("bench_out/protect_overhead_calibrated.json"))
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny model + few steps to validate the pipeline off-GPU",
    )
    args = p.parse_args(argv)

    if args.repeats < 5:
        print(f"WARNING: --repeats {args.repeats} < 5 violates design rule 4; "
              "the committed numbers must say so.")

    if args.smoke:
        args.n_layer, args.n_head, args.n_embd, args.block_size = 1, 2, 32, 16
        args.steps, args.orbits, args.repeats, args.warmup_rounds = 12, 2.0, 5, 1

    device = resolve_device(args.device)
    model_kw = dict(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        batch_size=args.batch_size, block_size=args.block_size,
    )
    ckpt_dir = args.json.parent / "poc-ckpt"

    # Conditions, in the fixed round-robin order. BASELINE and CONTROL are the
    # same code; every protected condition carries its rate.
    conditions = [BASELINE, CONTROL, *[f"protect@{r:.0e}" for r in CALIBRATED_RATES]]
    rate_of = {f"protect@{r:.0e}": r for r in CALIBRATED_RATES}

    print("=" * 78)
    print(f"protected-run wall-clock overhead (controlled) | device={device}")
    print(f"  model n_layer={args.n_layer} n_head={args.n_head} n_embd={args.n_embd} "
          f"block={args.block_size} | seed={args.seed} steps={args.steps} orbits={args.orbits}")
    print(f"  {args.repeats} interleaved repeats + {args.warmup_rounds} warmup, A/A control")
    print("=" * 78)

    walls: dict[str, list[float]] = {c: [] for c in conditions}
    # Last protected run per rate holds the (deterministic) recovery outcome.
    last: dict[str, tuple] = {}

    for rnd in range(args.warmup_rounds + args.repeats):
        timed = rnd >= args.warmup_rounds
        for c in conditions:
            if c in (BASELINE, CONTROL):
                wall = _time_baseline(device, model_kw, args.seed, args.steps)
            else:
                wall, r, rs, env = _time_protected(
                    device, model_kw, args.seed, args.steps, args.orbits,
                    rate_of[c], ckpt_dir,
                )
                last[c] = (r, rs, env)
            if timed:
                walls[c].append(wall)
        tag = "warmup" if not timed else f"repeat {rnd - args.warmup_rounds + 1}/{args.repeats}"
        print(f"  {tag} done")

    base = statistics.median(walls[BASELINE])
    control_ov = 100.0 * (statistics.median(walls[CONTROL]) - base) / base
    noise_floor = abs(control_ov)

    rows = []
    for c in conditions:
        if c in (BASELINE, CONTROL):
            continue
        w = statistics.median(walls[c])
        ov = 100.0 * (w - base) / base
        r, rs, env = last[c]
        rows.append(dict(
            rate=rate_of[c],
            protected_wall_s=round(w, 2),
            overhead_pct=round(ov, 1),
            resolvable=abs(ov) > noise_floor,
            rollbacks=rs["rollbacks"],
            replayed_steps=rs["replayed_steps"],
            checkpoints=rs["checkpoints_saved"],
            died=r.died,
            final_val=None if r.final_val_loss != r.final_val_loss else round(r.final_val_loss, 4),
            flips=env.stats.flips,
            flips_in_saa=env.stats.flips_in_saa,
            sched_in_mission=env.scheduled_within_mission,
        ))

    print("\n" + "=" * 78)
    print(f"{'rate':>10} {'wall_s':>9} {'overhead':>10} {'roll':>5} {'replay':>7} "
          f"{'val':>8} {'outcome':>10}")
    print("-" * 78)
    for row in rows:
        pct = "<noise" if not row["resolvable"] else f"{row['overhead_pct']:+.1f}%"
        outcome = "died" if row["died"] else "survived"
        val = "-" if row["final_val"] is None else f"{row['final_val']:.4f}"
        print(f"{row['rate']:>10.0e} {row['protected_wall_s']:>9.2f} {pct:>10} "
              f"{row['rollbacks']:>5} {row['replayed_steps']:>7} {val:>8} {outcome:>10}")
    print("-" * 78)
    print(f"{'baseline':>10} {base:>9.2f} {'--':>10}")
    print(f"{'A/A ctrl':>10} {'':>9} {noise_floor:>9.1f}%   <- resolution floor")
    print("=" * 78)

    out = dict(
        device=str(device),
        torch_version=torch.__version__,
        model=f"n_layer={args.n_layer} n_head={args.n_head} n_embd={args.n_embd} "
              f"block={args.block_size} batch={args.batch_size}",
        seed=args.seed,
        steps=args.steps,
        orbits=args.orbits,
        repeats=args.repeats,
        warmup_rounds=args.warmup_rounds,
        interleaved=True,
        aa_control=True,
        baseline_wall_s=round(base, 2),
        noise_floor_pct=round(noise_floor, 2),
        rows=rows,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2))
    print(f"baseline_wall_s {round(base, 2)}  noise_floor {noise_floor:.2f}%")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
