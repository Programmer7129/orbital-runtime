# PLAN.md — orbital-runtime MVP

**Source of truth for the builder session.** Read `docs/research/technical-foundations.md`
first — every number and tool choice below is sourced there. This file is self-contained;
the builder does not have access to the planning session's conversation.

## Mission (one paragraph of context)

We are building the YC-application demo for a startup thesis: commercial GPUs are going
to orbit by the thousands (Starcloud, Google Suncatcher, Axiom, SpaceX filings), cosmic
radiation flips their bits and crashes their jobs, and nobody sells the software layer
that makes cheap COTS GPUs survive it. The MVP proves the concept end-to-end in
simulation: a real PyTorch training run dies from radiation faults injected at published
LEO rates; the same run under our runtime survives with low overhead. Demo credibility
depends on the physics being *calibrated and citable*, not made up.

## The headline demo ("90 minutes in orbit, 90 seconds on screen")

A time-compressed simulated orbit. Split screen: two identical nanoGPT training runs on
one side, an orbital map + telemetry dashboard on the other. The satellite crosses the
South Atlantic Anomaly; the upset counter spikes; the UNPROTECTED run's loss curve
corrupts (NaN or silent divergence) and dies. The PROTECTED run detects the hits,
rolls back to a verified checkpoint, replays, and finishes — final banner shows:
`upsets injected / detected / recovered, wall-clock overhead X%`.

## Architecture

Python package `orbital_runtime/` (Python 3.11+, PyTorch ≥2.3):

```
orbital_runtime/
├── orbit/        # environment model
│   ├── track.py       # parametric 95-min LEO orbit; SAA window geometry (~10 min/orbit)
│   └── flux.py        # time-varying Poisson intensity λ(t): base rate × bits resident
│                      #   × SAA multiplier (50–100x) × storm multiplier (10–100x, off by default)
│                      #   modes: ecc_off | ecc_on (only multi-bit residuals + SEFIs leak)
├── inject/       # the fault injector (custom, ~300 lines core; PyTorchFI-style, vendored approach)
│   ├── memory.py      # Poisson-driven bit flips: tensor.view(torch.int32) ^= (1 << k)
│   │                  #   targets: params, optimizer state, (optionally) activations
│   ├── compute.py     # forward hooks corrupting random activation elements
│   ├── sefi.py        # simulated hangs/crashes (per-SAA-transit probability)
│   └── xid.py         # synthetic ECC/Xid event stream (Xid 48/63/64/94/95)
├── detect/
│   ├── guards.py      # tier 1 (free): isfinite on loss/grads; grad-norm z-score; loss-spike
│   ├── abft.py        # tier 2: sampled checksum verification around nn.Linear GEMMs
│   │                  #   variance-aware thresholds for bf16/fp16 (see V-ABFT in research doc)
│   └── watcher.py     # ECC/Xid consumer (synthetic in sim; DCGM/nvidia-smi on real NVIDIA)
├── ckpt/
│   ├── saver.py       # DCP async_save, double-buffered; model+optimizer+RNG+step
│   ├── policy.py      # orbit-aware cadence: checkpoint immediately before SAA entry;
│   │                  #   "adaptive vigilance" — ABFT sampling rate also keyed to λ(t)
│   └── recover.py     # detect → restore last VERIFIED checkpoint → replay
├── run.py        # CLI: `orbital-run --workload nanogpt --orbits 1 --rate 1e-8 --protect on|off`
└── telemetry.py  # JSONL event log (flip/detect/rollback/checkpoint) consumed by dashboard
demo/
├── workloads/nanogpt/   # char-level Shakespeare nanoGPT (small enough for CPU/MPS dev)
├── dashboard/           # web dashboard: orbit map, SAA shading, upset/detection counters,
│                        #   dual loss curves, overhead ticker (static build, reads JSONL live)
└── run_demo.sh          # reproduces the headline demo end-to-end
bench/
└── overhead.py          # measures per-tier overhead; emits results table for README/pitch
tests/                   # pytest; injector determinism (seeded), detector precision/recall,
                         # recovery correctness (bit-exact resume), λ(t) statistics
```

## Design rules

1. **Device-agnostic core.** Everything must run on CPU/MPS (dev happens on a Mac).
   CUDA-only features (DCGM/Xid polling, cuda-checkpoint) live behind interfaces with
   simulation fallbacks. Final headline numbers get produced on a rented cloud GPU.
2. **Calibration is sacred.** Default base rate sweepable 1e-9→1e-7 upsets/bit-day;
   every constant in `flux.py` carries a comment citing its source (see research doc).
   The demo must be able to say "these rates come from NASA NEPP / CREME96 / Suncatcher."
3. **Determinism.** Seeded runs must reproduce exactly (flip schedule, detection,
   recovery) — needed for tests, benchmarks, and a demo that never flakes on stage.
4. **Overhead honesty.** Report measured overhead per tier; target <10% total with
   ABFT sampling on. If we can't hit it, report what we hit — no fudging.

## Milestones (aggressive; YC deadline)

- **M0 — scaffold (days 1–2):** package layout, pyproject, pytest, CI-less but
  `make test`/`make demo` targets. Orbit model + Poisson engine with statistical tests
  (flip counts over N simulated orbits match λ(t) expectations; SAA share lands in the
  80–97% band from flight data).
- **M1 — break things (days 3–6):** injector complete; nanoGPT workload training on
  CPU/MPS; demonstrate both failure modes at elevated rates: (a) NaN/crash, (b) silent
  divergence (train-to-different-optimum, matching the AWS SDC paper's finding).
  Deliverable: `orbital-run --protect off` reliably produces a corrupted run.
- **M2 — see things (days 7–10):** all three detection tiers; measured precision/recall
  against known injected faults; overhead benchmark per tier.
- **M3 — survive things (days 11–14):** checkpoint policy + recovery loop; end-to-end
  protected run completes a multi-orbit session with injected faults; bit-exact resume
  test passes.
- **M4 — show things (days 15–18):** dashboard + demo script polish; rent an A100/H100
  (Lambda/RunPod) for headline numbers at realistic scale; record the demo video;
  README with results table + citations.
- **Stretch:** `cuda-checkpoint`+CRIU inference-service snapshot/restore demo;
  GoldenEye-style numeric-format experiments (bf16 vs fp32 vulnerability).

## What NOT to build (MVP discipline)

- No multi-node/distributed training support (single-node tells the story).
- No Gemini-style in-memory replication.
- No NVBitFI/SASS-level injection (cite as validation roadmap).
- No Kubernetes/orchestration integration (that's KubeSpace's layer, not ours).
- No real DCGM integration until the cloud-GPU week (M4) — simulated Xid until then.

## Acceptance criteria (demo-ready means)

1. `demo/run_demo.sh` produces the split-screen story on a laptop, deterministic, <5 min.
2. Unprotected run visibly fails; protected run completes; overhead <10% (or honestly
   reported); counters: injected ≥ detected ≥ recovered with precision/recall printed.
3. Every physics constant traceable to a citation in `docs/research/technical-foundations.md`.
4. Repo installs clean: `pip install -e . && make test` green on macOS and Linux/CUDA.
