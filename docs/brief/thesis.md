# Radiation-Induced Failure in Orbital GPU Compute — and the Software Layer That Survives It

**Technical thesis. Nothing simplified.** All figures cite primary sources; disclosed
limitations are marked. Companion: a one-command reproducible demo (see end).

---

## Thesis

Commercial GPUs are moving to orbit at scale. Cosmic radiation corrupts their
computations — not their transmissions — and no vendor-neutral software layer keeps a
COTS GPU *workload* alive through it. We build that layer: a fault-injection→detection→
recovery runtime, calibrated to flight-data upset rates and validated on datacenter
silicon. The moat is not the code (AI commoditizes that); it is beam-validated,
cross-generation failure data and flight heritage, accumulating per chip and per fleet.

## The problem is computation, not communication

Optical/RF downlinks are a solved, FEC-mature problem (NASA TBIRD: 200 Gbps, 4.8 TB
error-free through the atmosphere). The unsolved problem is bits corrupted *inside the
GPU while it computes*: single-event upsets (SEU), multi-bit upsets (MBU), functional
interrupts/crashes (SEFI), and cumulative-dose degradation (TID). ECC catches some
single-bit memory flips; nothing protects the training/inference job itself — corrupted
weights, poisoned optimizer state, silent divergence, or a hung device.

## Why it is worse than intuition suggests

- **Format-level lethality.** In fp32, for typical weights (|v|<1), flipping most
  exponent bits drives the value toward zero (harmless); flipping bit 30 multiplies by
  ~2¹²⁸ → NaN cascade. ~**1 in 32 bits is lethal**; a run absorbs ~100 upsets, then dies
  of one. Derived, not assumed — uniform-random injection; IEEE-754 decides.
- **Silent corruption, not just crashes.** Most failures announce themselves (NaN); a
  fraction finish with a **measurably worse model and zero alarms** — the case no
  transmission protocol or crash-monitor catches.
- **ECC shifts the threat, doesn't remove it.** On ECC-protected GPUs, detected-
  uncorrectable (crash) rate *exceeds* silent-corruption rate by **2.2–2.7×**
  [NSREC'21]; **31.5%** of HBM upsets are multi-bit and can evade single-bit ECC
  [MICRO'21]. The deployable (ECC-on) regime is dominated by exactly the channels with
  no published rates for modern accelerators.
- **The problem grows on two curves.** Each transistor shrink lowers critical charge →
  more upsets: H100 memory MTBE is **3.2× worse than A100** [NCSA, SC25]. And unit
  count in orbit rises yearly. Fragility × deployment, both monotonic.

## Why now

First datacenter-class GPU in orbit: Starcloud-1 (NVIDIA H100), Nov 2025. Google Project
Suncatcher (TPU clusters, Planet prototype ~2027). SpaceX orbital-AI constellation
filing. The rad-hard alternative is a non-starter for AI: the RAD750 (Perseverance,
~$200k/unit) is ~1998-class compute — roughly **10⁶× slower** than an H100. Every
operator flies commercial silicon and inherits the radiation problem. Rad-hardening
costs a million-fold compute penalty; our runtime costs **~2%**.

## What we built and measured

Device-agnostic (CPU/MPS/CUDA), deterministic (protected/unprotected runs face
bit-identical bombardment — a controlled experiment):

- **Calibrated injector.** Time-varying Poisson process over resident bits, band
  **1e-9…1e-7 upsets/bit-day** (NASA NEPP / CREME96 / flight-data anchored); SAA modeled
  as a phase-gated multiplier normalized to redistribute (not manufacture) daily upsets,
  reproducing the **80–97%** SAA-share seen in flight; MBU clustering per MICRO'21; SEFI
  calibrated to Suncatcher's measured **2×10⁻¹¹ cm²/chip**.
- **Three-tier detection.** Free guards (NaN/grad-norm/loss-spike); sampled ABFT
  checksums on GEMMs (Huang-Abraham 1984 lineage), sampling keyed to orbital position;
  ECC/Xid watcher against real hardware counters.
- **Orbit-aware recovery.** Async checkpoints cadenced to checkpoint before SAA entry;
  detect→restore→replay.

**Headline result (NVIDIA L4 24GB, CUDA, calibrated 1e-7 rate, no elevation):** an
85.3M-param training run **dies (NaN) at step 179** unprotected; the identical protected
run **completes 300/300** through 326 upsets and 10 rollbacks. Detection-only overhead
**+1.6%** (adaptive). Detection precision/recall measured with an exact determinism-based
oracle; **0/12** false positives on clean runs after variance-aware ABFT tolerance. 288
tests.

## Disclosed limitations (the honest edge)

ECC-off single-bit regime is the calibratable proxy; ECC-on residual channels
(MBU-leak/SRAM-logic/SEFI) carry conditions caveats from neutron/HBM2 source data. No
TID/aging term (out of scope for a rate model). SAA modeled as fixed-phase per orbit
(idealized). fp32 headline; bf16/fp16 differ. Two adversarial reviews (methodology +
artifact) were run against this work; findings fixed or disclosed, both in-repo.

## The open experiment

No published campaign irradiates a device **during training** and reports application-
level outcome distributions — verified unclaimed against the SEE literature (inference-
under-beam exists; training does not). A proton campaign (Crocker 67 MeV beamline — where
Google qualified its TPUs; Jetson-class DUT) would: (1) measure the beam-vs-simulation
agreement factor for tensor-level fault models of AI *training*; (2) calibrate the ECC-on
residual channels our model currently carries as literature-anchored assumptions. This is
the data no competitor can shortcut — it requires scarce beam time, radiation-effects
expertise, and it expires each chip generation.

## Why not just NVIDIA

NVIDIA ships primitives (cuda-checkpoint), not vertical reliability products — and never
built the fault-tolerant-training layer even for the vastly larger terrestrial market
(that came from ByteDance, Meta, Google, academia). Their Data Center revenue is $75.2B/
quarter [Q1 FY27]; orbital GPU volume through 2029 does not move that number. Their own
documentation states products are "not designed, authorized, or warranted to be suitable
for use in… space… equipment." When NVIDIA decides an ecosystem layer matters, they buy
the leader (Mellanox; Bright Computing; Run:ai, ~$700M, 2024). The plan is to be that
leader — holding cross-chip beam data, fleet telemetry, and flight heritage.

## Position

Resilience is the wedge: a privileged one, sitting beneath every workload on every node,
seeing every fault — the same launch position from which observability and security
platforms expanded from a single feature. Expansion surface: fleet observability,
health-aware scheduling, an assurance/insurance data business, cross-environment TAM
(LEO → cislunar → terrestrial edge/HPC), and a per-chip-generation re-characterization
subscription.

---

*Demo (one command, ~20 s, reproducible): unprotected training run dies from injected
flight-rate radiation; protected run survives. Repo + adversarial reviews available on
request. Contact: Vedant Patel — vedantspatel33@gmail.com*

*Sources: NASA NEPP/CREME96; NASA TBIRD; Google Project Suncatcher (arXiv:2511.19468);
MICRO'21 HBM2 soft errors (10.1145/3466752.3480111); NSREC'21 GPU DUE (arXiv:2108.00554);
NCSA "Story of Two GPUs" (arXiv:2503.11901); Huang & Abraham 1984; NVIDIA product
documentation & SLA; NVIDIA Q1 FY2027 results. Full citations in repo research docs.*
