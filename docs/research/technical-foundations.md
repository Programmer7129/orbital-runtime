# Technical Foundations (research synthesis, July 2026)

Ground truth for PLAN.md. Every design decision below traces to a source.

## 1. Fault injection — build our own, vendor the approach

- **PyTorchFI** (UIUC/NVIDIA, NCSA license): tensor-level injection via forward hooks.
  Semi-dormant (last push Jul 2024) — do NOT hard-depend; vendor/extend the approach.
  https://github.com/pytorchfi/pytorchfi
- **NVBitFI** (SASS-level, highest fidelity): stale since 2023, needs porting to modern
  NVBit. Cite as future validation path only. https://github.com/NVlabs/nvbitfi
- Tensor-level vs SASS-level injection don't perfectly match, but tensor-level is
  accepted in DSN/SC literature for error-propagation studies (arXiv 2412.08466).
- NVIDIA's own result: beam experiments and fault simulation agree within 5x — this
  legitimizes a simulation-first MVP. Cite: "Demystifying GPU Reliability" (NVIDIA, 2021).

**MVP approach:** ~300-line custom injector:
- Memory SEUs: Poisson process over resident bytes; `tensor.view(torch.int32) ^= (1 << k)`
  on random bits of params / optimizer state / activations.
- Compute SEUs: forward hooks corrupting random activation elements.
- SEFIs: simulated process hang/crash (Jetson RADECS 2024 finding: reboot cross-section
  EXCEEDS bit-error cross-section — crashes matter as much as flips).
- Synthetic ECC/Xid event stream (Xid 48/63/64/94/95) for the watcher tier.

## 2. Upset-rate calibration (the numbers that make the demo credible)

- COTS memory in LEO: 1e-3 to 1e-7 upsets/bit-day across generations; NASA NEPP design
  criteria 1e-5 to 1e-7; modern deep-submicron flight data at the low end (~1e-9 to 1e-8,
  Flying Laptop 600 km SSO). Methodology to cite: CREME96.
- **Sweepable base rate: 1e-9 to 1e-7 upsets/bit-day.** H100-class 80 GB HBM = 6.4e11 bits
  → 640–64,000 flips/day; at 1e-9 that's ~1 flip every 2 minutes (demo-friendly).
- **South Atlantic Anomaly (SAA):** flight data shows 80–97% of all LEO SEUs occur inside
  SAA transits (<20 min/orbit). Model as time-varying Poisson intensity with **50–100x
  multiplier ~10 min per ~95-min orbit** (Proba-II ~90%; 1025 km sat 97.3%).
- Storm mode: +10–100x transient (May 2024 Gannon storm, Wu 2025 Space Weather).
- ECC-on mode: only multi-bit residuals + logic faults + SEFIs leak through.
- Anchor citations for realism: Google Suncatcher (TPU HBM irregularities at 2 krad(Si),
  no hard failure to 15 krad, UC Davis Crocker cyclotron, arXiv 2511.19468); Jetson
  Xavier/Orin proton+TID beam papers (RADECS/IEEE 2021–2024); GPU HBM2 soft errors (MICRO'21).

## 3. Detection — three tiers

1. **Free tier (~0% overhead):** `torch.isfinite` on loss/grads per step; gradient-norm
   z-score + loss-spike detection (AWS "SDC in LLM training", arXiv 2502.12340: real SDCs
   cause loss spikes AND silent convergence drift); ECC/Xid log watcher (DCGM counters
   real on NVIDIA, synthetic in sim).
2. **ABFT tier (<10% overhead target):** checksum verification around nn.Linear GEMMs on
   a sampling schedule. Literature: FT-CNN 4–8%; V-ABFT ~12% w/ variance-based thresholds
   (solves fp16/bf16 rounding-noise-vs-fault discrimination, arXiv 2602.08043);
   ApproxABFT cuts exact-ABFT overhead ~43%.
3. **Escalation:** detection → recovery orchestrator.

**Differentiator: "adaptive vigilance"** — detection intensity and checkpoint cadence keyed
to orbital position (crank ABFT sampling + checkpoint immediately before SAA entry).
Novel; nothing in literature does position-aware protection scheduling.

## 4. Checkpoint / recovery

- **Training:** PyTorch Distributed Checkpoint (DCP) `async_save` (PyTorch ≥2.3),
  double-buffered to local NVMe: model + optimizer + RNG + step. Overhead precedent:
  CheckFreq ~3.5% (FAST'21). ByteCheckpoint (open source, active) if DCP hits limits.
- **Inference/process:** NVIDIA `cuda-checkpoint` (active, driver ≥550) + CRIU =
  transparent GPU process snapshot/restore (CRIUgpu, arXiv 2502.16631). Stretch demo.
- Recovery loop: detect → restore last VERIFIED checkpoint → replay steps.
- Do NOT build Gemini-style in-memory replication (SOSP'23) in MVP window.

## 5. Prior art / positioning

- **RedNet (arXiv 2407.11853):** per-layer selective protection for satellite DNN
  *inference*, model-specific, no public code. We are a general-purpose runtime for
  unmodified PyTorch training + inference. Cite; roadmap feature.
- **KubeSpace (arXiv 2601.21383):** container orchestration across satellites — the layer
  ABOVE us. Positioning line: "KubeSpace schedules containers across satellites; we keep
  the GPU workload inside the container alive through radiation."
- **Suncatcher:** proves COTS accelerators viable in orbit; explicitly leaves the software
  fault-tolerance layer open. That's the wedge.
- **GitHub whitespace confirmed:** no maintained "radiation-tolerant PyTorch runtime"
  exists — only FI research tools (pytorchfi, goldeneye, nvbitfi).

## Market context (from separate research pass, same session)

- No independent vendor sells chip-agnostic radiation fault-tolerance software for COTS
  datacenter GPUs. Starcloud hand-built theirs for one satellite; Aethero bundles theirs
  with their own hardware; Klepsydra targets ESA rad-hard processors, not COTS GPUs.
- YC Summer 2026 RFS (authored by Starcloud's CEO) explicitly calls for space compute
  "optimized for mass, thermal performance, and radiation."
- Target customers (tier-2 operators): Sophia Space, Madari Space, Orbital (a16z),
  Aethero, Little Place Labs, Axiom ecosystem; dual-use terrestrial angle: silent data
  corruption in ground GPU fleets (Meta/Google SDC literature).
