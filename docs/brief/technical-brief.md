# Surviving Radiation: A Fault-Tolerance Runtime for Commercial GPUs in Orbit

**Technical brief for academic review — Vedant Patel, July 2026**
*(10-minute read. Companion artifacts: one-command demo, full source, and two adversarial
methodology reviews, available on request.)*

## The problem

Commercial GPUs are being launched into orbit at accelerating scale — Starcloud flew the
first H100 in late 2025, Google's Project Suncatcher targets TPU clusters by 2027, and
SpaceX has filed for an orbital-AI constellation. Google's Suncatcher paper characterized
the *hardware* radiation response of its accelerators (TID/SEE proton testing at UC Davis
Crocker Nuclear Laboratory) but explicitly left the *software* fault-tolerance layer open.
No published runtime keeps an unmodified PyTorch training job alive through single-event
upsets on COTS accelerators. We built one, and we want to validate its fault model against
beam ground truth.

## What we built

`orbital-runtime`: a Python/PyTorch runtime with three coupled subsystems, all
device-agnostic (CPU/MPS/CUDA), fully deterministic (named RNG streams; protected and
unprotected runs face bit-identical bombardment — a controlled experiment, not a vibe):

1. **Calibrated fault injection** — a time-varying Poisson process over resident model
   bits (params + optimizer state), rate band 1e-9–1e-7 upsets/bit-day anchored to NASA
   NEPP guidance, CREME96 methodology, and recent flight data; South Atlantic Anomaly
   modeled as a phase-gated intensity multiplier normalized to *redistribute* (not
   manufacture) daily upsets, reproducing the 80–97% SAA share seen in flight data.
2. **Three-tier detection** — near-free guards (isfinite / gradient-norm z-score /
   loss-spike), sampled ABFT checksums on GEMM weights, and an ECC/Xid watcher
   (implemented against real NVIDIA counters on hardware, simulated otherwise).
3. **Orbit-aware checkpoint/recovery** — asynchronous double-buffered checkpoints with
   cadence keyed to orbital position (checkpoint before SAA entry; detection sampling
   concentrated where ~90% of upsets land), plus rollback-and-replay recovery.

## Headline result (measured on a rented NVIDIA L4 24GB, CUDA)

At the **calibrated top-of-band rate (1e-7 upsets/bit-day, no artificial elevation)** on
an 85.3M-parameter GPT-style training run: the unprotected run **dies (NaN) at step 179**;
the identical protected run **completes 300/300 steps** through 326 upsets and 10
rollbacks. Detection-only overhead on the same hardware: **+1.6%** with position-aware
adaptive sampling (vs +5.4% at 100% sampling). Full result tables, confidence intervals,
and raw JSON artifacts are in the repository; every physics constant carries a source
citation in code.

One mechanistic finding we believe is not in the literature: for fp32 weights with
|v|<1, only exponent bit 30 is catastrophic (sets the exponent high; other exponent-bit
flips drive values harmlessly toward zero) — so training runs absorb ~100 upsets and then
die of a single specific bit, which our seed-level data reproduces exactly.

## What we know is weak — and why that is the collaboration

We commissioned two independent adversarial reviews (methodology and
artifact-evaluation; both in the repo, findings fixed or disclosed). The surviving
limitations are precisely the questions simulation cannot close:

- **Threat-model inversion.** Our quantitative headline is the ECC-off single-bit-DRAM
  regime — the calibratable proxy. Real deployments fly ECC-on, where the residual
  channels (multi-bit-upset leakage, SRAM/register/logic upsets, SEFIs) dominate — and
  those are exactly the channels for which no public rates exist for modern accelerators.
- **Injection realism.** We inject at tensor level, between optimizer steps, into
  params/optimizer state only. Whether that faithfully proxies hardware fault
  manifestation (mid-kernel corruption, gradient-path upsets, MBU spatial patterns) is
  an open empirical question. NVIDIA's beam-vs-simulation work found ~5x agreement for
  SASS-level models; no equivalent exists for tensor-level injection or for AI-training
  workloads.
- **Rate provenance.** Unprotected death is demonstrated at the top of the cited band;
  flight data for modern deep-submicron memory clusters toward 1e-9–1e-8, where
  unprotected runs survive our tested mission lengths.

## The proposed experiment

A proton-beam campaign at Crocker (67 MeV, the Suncatcher beamline) with a Jetson-class
COTS module (we supply hardware) running our instrumented workload:

1. Measure application-level fault manifestation (weight corruption, silent divergence,
   SEFI/hang rates) under known fluence.
2. Compare against our injector's predictions at matched effective rates —
   the "beam-vs-simulation agreement factor" for tensor-level fault models of AI
   workloads, which does not exist in the literature.
3. Calibrate the ECC-on residual channels (MBU fraction, SEFI per-fluence probability)
   that our model currently carries as explicitly-uncited placeholders.

Deliverables: a co-authored paper (we believe this is publishable at NSREC/RADECS or
DSN), an openly calibrated fault model for the community, and — our commercial interest,
stated plainly — a validated foundation for a reliability runtime we intend to
commercialize for the orbital-compute industry.

## Why review this

The engineering hygiene is unusual for a startup artifact: A/A-controlled benchmarks,
determinism-pinned experiments, Clopper-Pearson intervals on every ratio, self-documented
modeling bugs, and adversarial reviews committed to the repo. The demo reproduces in one
command in ~20 seconds on a laptop. We are asking for 30 minutes and a skeptical eye.

*Contact: Vedant Patel — vedantspatel33@gmail.com*
