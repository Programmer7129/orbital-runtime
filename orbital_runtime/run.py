"""CLI: `orbital-run --workload nanogpt --orbits 1 --rate 1e-8 --protect on|off`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .ckpt.policy import CheckpointPolicy
from .ckpt.recover import RecoveryOrchestrator
from .ckpt.saver import CheckpointSaver
from .detect import Detector, GuardTier
from .detect.abft import AbftTier
from .detect.watcher import SimulatedXidSource, WatcherTier
from .inject.injector import RadiationEnvironment
from .inject.sefi import SefiInjector
from .inject.xid import XidSimulator
from .rng import STREAM_ABFT, stream
from .orbit.flux import (
    DEFAULT_BASE_RATE_UPSETS_PER_BIT_DAY,
    DEFAULT_SAA_MULTIPLIER,
    MODE_ECC_OFF,
    MODE_ECC_ON,
    FluxModel,
)
from .orbit.track import OrbitTrack
from .telemetry import Telemetry
from .train import TrainConfig, train
from .workload import get_workload, resolve_device

DEFAULT_RUN_DIR = Path("runs")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orbital-run",
        description="Run a PyTorch workload through a simulated LEO radiation environment.",
    )
    p.add_argument("--workload", default="nanogpt", choices=["nanogpt"])
    p.add_argument("--orbits", type=float, default=1.0, help="simulated orbits to cover")
    p.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_BASE_RATE_UPSETS_PER_BIT_DAY,
        help="base upset rate, upsets/bit-day (sweep 1e-9..1e-7; 0 disables radiation)",
    )
    p.add_argument("--protect", choices=["on", "off"], default="off")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    p.add_argument("--tag", default="run", help="label for the telemetry file")
    p.add_argument("--out", type=Path, default=DEFAULT_RUN_DIR)

    g = p.add_argument_group("environment")
    g.add_argument("--saa-multiplier", type=float, default=DEFAULT_SAA_MULTIPLIER)
    g.add_argument("--ecc", choices=[MODE_ECC_OFF, MODE_ECC_ON], default=MODE_ECC_OFF)
    g.add_argument("--storm", action="store_true", help="enable storm mode (10-100x)")
    g.add_argument(
        "--sefi-prob",
        type=float,
        default=0.0,
        help="per-SAA-transit SEFI probability (uncalibrated; default off)",
    )
    g.add_argument(
        "--inject-activations",
        action="store_true",
        help="also corrupt activations via forward hooks",
    )
    g.add_argument(
        "--activation-share",
        type=float,
        default=0.5,
        help="fraction of upsets routed to activations when enabled",
    )

    d = p.add_argument_group("protection (--protect on)")
    d.add_argument("--no-abft", action="store_true", help="tier 1 only")
    d.add_argument("--abft-rate", type=float, default=0.1, help="ABFT sampling outside SAA")
    d.add_argument("--abft-saa-rate", type=float, default=1.0, help="ABFT sampling in SAA")
    d.add_argument("--ckpt-interval", type=int, default=50, help="checkpoint cadence, steps")
    d.add_argument("--ckpt-saa-interval", type=int, default=10, help="cadence inside SAA")
    d.add_argument(
        "--no-adaptive",
        action="store_true",
        help="disable orbit-aware vigilance (uniform sampling + cadence)",
    )
    d.add_argument("--sync-checkpoint", action="store_true", help="disable async DCP save")

    w = p.add_argument_group("workload")
    w.add_argument("--batch-size", type=int, default=16)
    w.add_argument("--block-size", type=int, default=64)
    w.add_argument("--n-layer", type=int, default=4)
    w.add_argument("--n-head", type=int, default=4)
    w.add_argument("--n-embd", type=int, default=128)
    w.add_argument("--lr", type=float, default=1e-3)
    w.add_argument("--eval-every", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)

    workload = get_workload(
        args.workload,
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        lr=args.lr,
    )

    run_id = f"{args.tag}-s{args.seed}"
    log_path = Path(args.out) / f"{run_id}.jsonl"
    telemetry = Telemetry(path=log_path, run_id=run_id, tag=args.tag)

    env = None
    flux = None
    if args.rate > 0:
        # Resident bits come from the workload itself, so lambda is scaled to
        # what is actually there to be hit -- not to a hypothetical device.
        from .inject.memory import MemoryInjector

        probe = MemoryInjector(workload.model, workload.optimizer)
        bits = probe.static_resident_bits()

        flux = FluxModel(
            bits_resident=bits,
            track=OrbitTrack(),
            base_rate_upsets_per_bit_day=args.rate,
            saa_multiplier=args.saa_multiplier,
            storm_enabled=args.storm,
            mode=args.ecc,
        )
        env = RadiationEnvironment(
            workload.model,
            workload.optimizer,
            flux=flux,
            seed=args.seed,
            n_steps=args.steps,
            orbits=args.orbits,
            telemetry=telemetry,
            sefi=SefiInjector(flux.track, p_per_transit=args.sefi_prob),
            xid=XidSimulator(ecc_on=(args.ecc == MODE_ECC_ON)),
            inject_activations=args.inject_activations,
            activation_share=args.activation_share,
        )

    detector = None
    recovery = None
    if args.protect == "on":
        abft = (
            AbftTier(
                workload.model,
                flux=flux,
                base_sample_rate=args.abft_rate,
                saa_sample_rate=args.abft_saa_rate,
                adaptive=not args.no_adaptive,
                rng=stream(args.seed, STREAM_ABFT),
            ).attach()
            if not args.no_abft
            else None
        )
        watcher = None
        if env is not None:
            watcher = WatcherTier(source=SimulatedXidSource(env.xid))
        detector = Detector(guards=GuardTier(), abft=abft, watcher=watcher)

        saver = CheckpointSaver(
            workload.model,
            workload.optimizer,
            directory=Path(args.out) / f"{run_id}-ckpt",
            use_async=not args.sync_checkpoint,
        )
        recovery = RecoveryOrchestrator(
            saver=saver,
            policy=CheckpointPolicy(
                track=OrbitTrack(),
                base_interval=args.ckpt_interval,
                saa_interval=args.ckpt_saa_interval,
                adaptive=not args.no_adaptive,
            ),
            env=env,
            telemetry=telemetry,
            detector=detector,
        )

    _print_header(args, workload, device, flux, env)

    cfg = TrainConfig(steps=args.steps, eval_every=args.eval_every)
    result = train(
        workload, cfg=cfg, env=env, telemetry=telemetry, detector=detector, recovery=recovery
    )

    if env is not None:
        env.close()
    telemetry.close()

    print("\n" + "=" * 72)
    print(result.summary())
    if env is not None:
        s = env.stats
        print(
            f"  upsets delivered {s.flips} ({s.flips_in_saa} in SAA) | "
            f"non-finite weights {s.flips_nonfinite} | "
            f"activation hits {s.activation_hits} | SEFIs {s.sefis} | Xids {s.xids}"
        )
        if env.schedule_exhausted:
            print(
                "  WARNING: the run outlived its drawn radiation schedule; "
                "the tail of this run was unirradiated. Do not quote it."
            )
    if detector is not None:
        st = detector.stats()
        print(
            f"  detections {st['detections']} by tier {st.get('detections_by_tier', {})} | "
            f"ABFT verified {st.get('abft_gemms_verified', 0)}/{st.get('abft_gemms_seen', 0)} "
            f"GEMMs ({st.get('abft_sample_rate_actual', 0)*100:.0f}%)"
        )
    if recovery is not None:
        rs = recovery.stats_dict()
        print(
            f"  checkpoints {rs['checkpoints_saved']} "
            f"(pre-SAA {rs['pre_saa_saves']}, interval {rs['interval_saves']}) | "
            f"rollbacks {rs['rollbacks']} | replayed {rs['replayed_steps']} steps | "
            f"rejected {rs['checkpoints_rejected']}"
        )
    print(f"  telemetry: {log_path}")
    print("=" * 72)

    return 1 if result.died else 0


def _print_header(args, workload, device, flux, env) -> None:
    n_params = sum(p.numel() for p in workload.model.parameters())
    print("=" * 72)
    print(f"orbital-run | workload={args.workload} device={device} seed={args.seed}")
    print(f"  model: {n_params/1e6:.2f}M params | steps={args.steps} orbits={args.orbits}")
    if flux is None:
        print("  radiation: OFF (clean baseline)")
        return
    print(
        f"  radiation: rate={args.rate:.1e} upsets/bit-day | mode={args.ecc}"
        f"{' | STORM' if args.storm else ''}"
    )
    print(
        f"  resident bits: {flux.bits_resident:.3e} | expected {flux.expected_upsets_per_day():.1f} upsets/day"
    )
    print(
        f"  SAA: {flux.track.saa_duration_s/60:.0f} min per {flux.track.period_s/60:.0f} min orbit"
        f" | {args.saa_multiplier:.0f}x | share {flux.saa_share()*100:.1f}%"
    )
    if env is not None:
        print(f"  scheduled upsets over mission: {env.scheduled_within_mission}")
    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
