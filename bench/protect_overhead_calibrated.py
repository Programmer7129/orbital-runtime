"""Protected-run WALL-CLOCK overhead at CALIBRATED rates (handoff SS2).

Never measured before: the M2 overhead table is detection-only (radiation
OFF) and excludes replay. This measures the full wall-clock cost of
`--protect on` -- detection + checkpoint saves + any rollback/replay --
against a clean baseline, at each calibrated flight-band rate. At these
rates rollbacks are rare, so the number should sit near the detection-only
figure; that is the point.
"""
import json, statistics, time
import torch
from orbital_runtime.workload import get_workload, resolve_device
from orbital_runtime.inject.injector import RadiationEnvironment
from orbital_runtime.inject.memory import MemoryInjector
from orbital_runtime.inject.xid import XidSimulator
from orbital_runtime.orbit.flux import FluxModel
from orbital_runtime.orbit.track import OrbitTrack
from orbital_runtime.train import TrainConfig, train
from orbital_runtime.detect import Detector, GuardTier
from orbital_runtime.detect.abft import AbftTier
from orbital_runtime.detect.watcher import SimulatedXidSource, WatcherTier
from orbital_runtime.ckpt.policy import CheckpointPolicy
from orbital_runtime.ckpt.recover import RecoveryOrchestrator
from orbital_runtime.ckpt.saver import CheckpointSaver
from orbital_runtime.rng import STREAM_ABFT, stream
from pathlib import Path

dev = resolve_device("cuda")
MODEL = dict(n_layer=12, n_head=12, n_embd=768, batch_size=16, block_size=256)
SEED = 3
STEPS = 150
ORBITS = 4.0
REPEATS = 2

def _sync():
    if dev.type == "cuda": torch.cuda.synchronize()

def time_baseline():
    w = get_workload("nanogpt", seed=SEED, device=dev, **MODEL)
    t0=time.perf_counter(); train(w, cfg=TrainConfig(steps=STEPS)); _sync()
    return time.perf_counter()-t0

def time_protected(rate):
    w = get_workload("nanogpt", seed=SEED, device=dev, **MODEL)
    bits = MemoryInjector(w.model, w.optimizer).static_resident_bits()
    flux = FluxModel(bits_resident=bits, track=OrbitTrack(), base_rate_upsets_per_bit_day=rate)
    xid = XidSimulator(ecc_on=False)
    env = RadiationEnvironment(w.model, w.optimizer, flux=flux, seed=SEED, n_steps=STEPS, orbits=ORBITS, xid=xid)
    abft = AbftTier(w.model, flux=flux, base_sample_rate=0.1, saa_sample_rate=1.0, adaptive=True, rng=stream(SEED, STREAM_ABFT)).attach()
    det = Detector(guards=GuardTier(), abft=abft, watcher=WatcherTier(source=SimulatedXidSource(env.xid)))
    saver = CheckpointSaver(w.model, w.optimizer, directory=Path(f"runs/ovh-ckpt"), use_async=True)
    rec = RecoveryOrchestrator(saver=saver, policy=CheckpointPolicy(track=OrbitTrack(), base_interval=50, saa_interval=10, adaptive=True), env=env, detector=det)
    t0=time.perf_counter(); r=train(w, cfg=TrainConfig(steps=STEPS), env=env, detector=det, recovery=rec); _sync()
    wall=time.perf_counter()-t0
    rs=rec.stats_dict()
    return wall, r, rs, env

base = statistics.median([time_baseline() for _ in range(REPEATS)])
rows=[]
for rate in (1e-9,1e-8,1e-7):
    walls=[]; last=None
    for _ in range(REPEATS):
        wall, r, rs, env = time_protected(rate); walls.append(wall); last=(r,rs,env)
    w = statistics.median(walls)
    r,rs,env=last
    ov=100.0*(w-base)/base
    rows.append(dict(rate=rate, protected_wall_s=round(w,2), overhead_pct=round(ov,1),
        rollbacks=rs["rollbacks"], replayed_steps=rs["replayed_steps"], checkpoints=rs["checkpoints_saved"],
        died=r.died, final_val=None if r.final_val_loss!=r.final_val_loss else round(r.final_val_loss,4),
        flips=env.stats.flips, flips_in_saa=env.stats.flips_in_saa, sched_in_mission=env.scheduled_within_mission))
    print(rows[-1])
out=dict(device=str(dev), model="85M n_layer=12 n_embd=768 block=256", seed=SEED, steps=STEPS, orbits=ORBITS, repeats=REPEATS, baseline_wall_s=round(base,2), rows=rows)
Path("bench_out").mkdir(exist_ok=True)
Path("bench_out/protect_overhead_l4.json").write_text(json.dumps(out, indent=2))
print("baseline_wall_s", round(base,2))
print("wrote bench_out/protect_overhead_l4.json")
