# orbital-runtime

**A software runtime that makes commercial GPUs survive radiation-induced faults in orbit.**
Calibrated fault injection at published LEO upset rates → three-tier detection → orbit-aware
checkpointing → invisible job recovery. *"ECC memory for the orbital era, sold as software."*

Commercial GPUs are going to orbit by the thousands (Starcloud, Google Suncatcher, Axiom,
SpaceX filings). Cosmic radiation flips their bits and crashes their jobs, and no independent
vendor sells the software layer that keeps a COTS GPU workload alive through it. This repo
proves the concept end-to-end **in simulation**: a real PyTorch training run dies from
single-event upsets injected at flight-data rates; the *same* run under this runtime survives
with low overhead. Every number in the physics is calibrated and citable, not invented.

---

## The demo — "90 minutes in orbit, 90 seconds on screen"

```bash
make venv install     # one-time: create .venv, install the package + dev deps
demo/run_demo.sh       # trains 3 runs, builds the dashboard, opens it
```

`run_demo.sh` trains three **identical** seeded nanoGPT runs that differ only in radiation and
protection, then opens a split-screen mission-control dashboard
(`demo/dashboard/index.html` — self-contained local HTML/JS, no server, no CDN):

- an **orbit + SAA** view (satellite crossing the South Atlantic Anomaly, upsets flashing),
- **live counters** (upsets injected / detected / rolled back / steps replayed),
- **dual loss curves** (protected vs unprotected vs clean baseline), and
- a scrolling **recovery event log** + wall-clock overhead ticker.

The unprotected run's loss looks perfectly healthy — then a bit flip NaNs it. The protected run
takes the same bombardment, detects the strikes that would propagate, rolls back to a verified
checkpoint each time, replays, and **finishes.** The unprotected run dies; the protected run lives.

The **committed dashboard** (`demo/dashboard/telemetry_data.js`) is the **NVIDIA L4 calibrated-rate
mission** (seed 3, 300 steps, 1e-7/bit-day — no elevation; unprotected dies at step 179, protected
completes 300/300; see the real-scale results below). `run_demo.sh` is the **no-GPU reproducer**: on
a Mac it retrains a laptop-scale version (seed 1337, 0.81M params, compensated 3e-6 rate — unprotected
dies ~step 141) in **under a minute**, deterministically. Both stories are the same mechanism at two
scales; the honest disclosure of the laptop rate is in `run_demo.sh` and the honesty flags.

---

## Headline results

### Real scale — NVIDIA L4 24GB, at the CALIBRATED flight-band rate (M4b)

Re-measured on a rented **NVIDIA L4 24GB** (ECC on), an **85.3M-param GPT-2-class nanoGPT**
holding **8.19e9 resident bits** (~100× the laptop demo model). At this scale the **calibrated
1e-7 upsets/bit-day rate — the top of the flight band, with no elevation at all — kills the
unprotected run on its own.** The demo no longer needs a compensated rate.

**Calibrated-rate mission (seed 3, 300 steps, 4 orbits, rate 1e-7, NVIDIA L4):**

| Run | Radiation | Outcome | Final val loss |
|---|---|---|---|
| clean baseline | none | completed 300/300 | **2.5160** |
| `--protect off` | 1e-7/bit-day (calibrated) | **DIED (NaN) at step 179** | ∞ |
| `--protect on` | 1e-7/bit-day (calibrated) | **COMPLETED 300/300** | **2.6275** |

Unprotected: 128 upsets (114 in SAA), then NaN death. Protected: absorbed **326 upsets** (302 in
SAA — it flies a longer, replayed mission), **10 detected → 10 rolled back** (8 ABFT · 2 guard),
141 steps replayed, and finished. The committed dashboard (`demo/dashboard/telemetry_data.js`) is
this exact run.

**Detection-only overhead** (radiation off — the cost of *looking*), **NVIDIA L4, 85.3M params**:

| Config | NVIDIA L4 24GB (85.3M) |
|---|---|
| A/A control (noise floor) | 0.4% |
| tier 1 guards | below noise |
| tier 1 + 2, **adaptive** sampling | **+1.6% ✓** |
| tier 1 + 2 @ 100% sampling | +5.4% ✓ |

Better than the MPS figure, exactly as predicted: ABFT is kernel-launch-bound and the cost
amortizes as GEMMs grow — at real scale even **100% sampling meets the <10% target**.

**Protected-run WALL-CLOCK overhead at calibrated rates** (incl. DCP checkpoint I/O + replay —
*never measured before M4b*), **NVIDIA L4** (seed 3, 150 steps, 4 orbits, 2 repeats, baseline 53.2 s):

| Rate (upsets/bit-day) | Wall-clock overhead | Rollbacks | Steps replayed | Outcome |
|---|---|---|---|---|
| 1e-9 | +27.9% | 1 | 10 | survived |
| 1e-8 | +64.0% | 3 | 20 | survived |
| 1e-7 | +139.6% | 9 | 64 | survived |

This is the honest full-cost number the M2 table excludes. It is dominated **not by detection
(+1.6%) but by full-model checkpoint I/O + replay** — the genuine price of turning the unprotected
run's death into survival. Checkpoint cadence is a tunable knob; the replay share scales with how
much real corruption actually landed.

**Detector precision / recall at demo scale** (85.3M params, calibrated 1e-7, 6 seeds), **NVIDIA L4**:

| Tiers | Precision (irradiated) | Recall | Median latency |
|---|---|---|---|
| tier 1 guards | 1.00 | 1.00 | 9 steps |
| tier 1 + 2 (guards + ABFT) | 1.00 | 1.00 | **4 steps** |

Recall holds at 1.00 at real scale, ABFT-driven (first detection is `abft_mismatch` on 6/6);
6/6 irradiated runs corrupted. ⚠️ **A scale-dependent limitation surfaced** — ABFT false-positives
on **3/6 *clean* runs** at 768-dim (see honesty flags), so clean-run precision does *not* yet hold
at real scale without a cancellation-aware tolerance. (The demo seed 3 is unaffected: 0 clean FP.)

Raw JSON for all four: `bench/results/*_l4.json`.

### Laptop demo (no GPU) — seed 1337, 0.81M-param nanoGPT, 200 steps, 2 orbits, MPS/CPU

| Run | Radiation | Outcome | Final val loss |
|---|---|---|---|
| clean baseline | none | completed 200/200 | **2.4304** |
| `--protect off` | 3e-6/bit-day | **DIED (NaN) at step 141** | ∞ |
| `--protect on` | 3e-6/bit-day | **COMPLETED 200/200** | **2.4288** |

Protected run: **49 upsets absorbed, 7 detected** (5 ABFT · 2 guard), **7 rollbacks, 105 steps
replayed**, ABFT sampling 19.1% average. Recovered a model **indistinguishable from the run that
was never irradiated** (2.4288 vs 2.4304, +0.07%).

> ⚠️ **The 3e-6 rate is ~300× the calibrated flight band, and this is disclosed everywhere it
> appears.** The demo model holds 7.8e7 resident bits against an H100's 6.4e11 — four orders of
> magnitude fewer bits to hit — so a flight-band rate delivers almost nothing in 200 steps. The
> *rate band itself* is asserted against real H100 bit counts in `tests/test_flux.py`; the
> per-tier overhead below is measured at the true rate. **Headline numbers at real scale are M4b
> (a rented GPU), not this laptop.**

### Detection — precision / recall vs known injected faults (tiny model, 12 seeds, CPU)

*(Tiny 1-layer/32-dim model; the L4 demo-scale re-measurement is in the real-scale section above.)*

| Tiers | Precision | Recall | Median latency |
|---|---|---|---|
| tier 1 (finite/z-score/loss-spike guards) | 1.00 | 0.83 | 24 steps |
| tier 1 + 2 (guards + ABFT) | **1.00** | **1.00** | **2 steps** |

Zero false positives across 12 clean runs *at this scale* (at 768-dim, ABFT false-positives on
clean runs — see honesty flags). Ground truth is *exact*, not thresholded:
determinism gives a free oracle — the first step where a rate-0 and an irradiated run's losses
diverge is, by construction, the first fault that **propagated**, so recall is measured only
against faults that can actually hurt the model (a ReLU-masked upset is never scored as a miss).

### Overhead per tier — dev machine MPS/CPU (measured, with an A/A noise-floor control)

*(Dev-machine MPS/CPU; the NVIDIA L4 detection-only overhead is +1.6% adaptive — real-scale section above.)*

| Config | CPU (0.81M) | MPS (0.81M) | MPS (10.7M) |
|---|---|---|---|
| A/A control (noise floor) | 3.2% | 0.8% | 0.3% |
| tier 1 guards | below noise | below noise | below noise |
| tier 1 + 2, **adaptive** sampling | **+7.2% ✓** | +22.3% | **+5.4% ✓** |
| tier 1 + 2 @ 100% sampling | +15.0% | +127.6% | +36.3% |

Tier 1 is genuinely free. The **<10% target is met at scale** (+5.4% on 10.7M params) precisely
because of **adaptive vigilance**: ABFT sampling is keyed to orbital position, covering the SAA —
where ~90% of upsets land — completely, at 19% *average* sampling. 100% sampling never meets the
target; that gap is the quantitative case for position-aware protection.

### Environment calibration (H100-class, 6.4e11 resident bits)

| Base rate (upsets/bit-day) | Upsets/day | Mean interval | SAA share |
|---|---|---|---|
| 1e-9 | 640 | 135 s | 89.8% |
| 1e-8 (default) | 6,400 | 13.5 s | 89.8% |
| 1e-7 | 64,000 | 1.4 s | 89.8% |

The 89.8% SAA share (default 75× multiplier) lands inside the **80–97% band from flight data**
and reproduces Proba-II's observed ~90%. Daily total is invariant to the multiplier — the SAA
*redistributes* upsets in time, it does not manufacture them.

---

## Architecture

```
orbital_runtime/
├── orbit/     track.py  — parametric 95-min LEO orbit; phase-gated SAA window (~10 min/orbit)
│              flux.py   — time-varying Poisson intensity λ(t): base × bits × SAA × storm
├── inject/    memory.py — Poisson bit-flips: tensor.view(int32) ^= (1<<k) on params/optimizer
│              compute.py, sefi.py, xid.py — activation hooks, hangs/crashes, synthetic Xid stream
├── detect/    guards.py — tier 1: isfinite + grad-norm z-score + loss-spike (≈free)
│              abft.py   — tier 2: sampled checksum verification around nn.Linear GEMMs
│              watcher.py— tier 3: ECC/Xid consumer (synthetic in sim; real nvidia-smi ECC on L4)
├── ckpt/      saver.py, policy.py, recover.py — DCP checkpoints; orbit-aware cadence; detect→restore→replay
├── run.py     — CLI: orbital-run --workload nanogpt --orbits 2 --rate 3e-6 --protect on|off
└── telemetry.py — JSONL event log (the single source of truth the dashboard reads)
demo/
├── workloads/nanogpt/  — char-level Shakespeare nanoGPT (CPU/MPS-sized)
├── dashboard/          — self-contained HTML/JS dashboard + build.py (JSONL → telemetry_data.js)
└── run_demo.sh         — the headline demo, end to end
bench/  overhead.py, detect_eval.py, results/*_l4.json   tests/  (263, green on macOS/MPS + Linux/CUDA)
```

**Differentiator — "adaptive vigilance":** detection intensity *and* checkpoint cadence are keyed
to orbital position (crank ABFT sampling + checkpoint immediately before SAA entry). Nothing in
the literature does position-aware protection scheduling.

### Design rules (enforced, not aspirational)

1. **Device-agnostic core** — runs on CPU/MPS; CUDA-only paths (DCGM/Xid, cuda-checkpoint) sit
   behind interfaces with simulation fallbacks.
2. **Calibration is sacred** — every physics constant in `flux.py`/`track.py` carries a comment
   citing its source. (Two remaining exceptions are flagged below.)
3. **Determinism** — seeded runs reproduce exactly: the flip schedule is drawn before step 0 from
   named RNG streams, so protected and unprotected runs face a **bit-identical bombardment** and
   the comparison is a controlled experiment. Turning ABFT on cannot shift the radiation.
4. **Overhead honesty** — overhead is measured against an A/A control; effects below the noise
   floor are reported as "below noise", never as a number.

---

## Running it yourself

```bash
make venv install     # .venv + editable install + dev deps
make test              # full suite: 262 pass on macOS/MPS (261 on Linux/CUDA), ~30-40 s
make demo              # the three runs, raw (no dashboard)
demo/run_demo.sh       # the three runs + dashboard (NO_OPEN=1 to skip auto-open)

# sweep the knobs directly:
.venv/bin/python -m orbital_runtime.run --rate 1e-8 --protect on --orbits 3 --steps 400
```

Regenerate the dashboard from existing logs without retraining:
`python demo/dashboard/build.py`.

---

## Honesty flags (what is *not* yet proven)

These are tracked in `STATUS.md` and enforced in code; nothing here is quietly shipped.

- **The laptop demo rate (3e-6) is model-size compensation, not physics** — disclosed in the
  Makefile, `run_demo.sh`, and above. It exists only because the 0.81M laptop model has too few bits
  to hit at flight rates. **On the L4 this is retired: the calibrated 1e-7 rate kills the unprotected
  run with no elevation** (see the real-scale section). The 1e-9→1e-7 band is what real H100 bit
  counts imply, asserted in `tests/test_flux.py`.
- **ABFT false-positives at real scale (found M4b, not yet fixed).** On the 85.3M model, ABFT's
  checksum trips on **3/6 *clean* (unirradiated) runs** (`abft_mismatch`, a *certain* verdict) — it
  was 0/12 at the 32-dim test scale. Root cause: the mismatch tolerance scales with the
  post-reduction `|value|`, but the checksum sums over the wide output dimension, so **catastrophic
  cancellation** makes true fp32 rounding noise exceed the tolerance on some steps. The fix is a
  running-error / L1 tolerance bound — the "variance-aware threshold" of V-ABFT the module already
  cites (`detect/abft.py`); **recall is unaffected** (real faults dwarf any tolerance, so the
  detect_eval recall stays 1.00). Until then, clean-run *precision* is a tiny-scale result. The
  headline mission (seed 3) is unaffected — 0 clean FP on that seed.
- **Two uncited constants, both OFF by default** — `DEFAULT_ECC_LEAK_FRACTION` (`flux.py`) and the
  SEFI `p_per_transit` (`sefi.py`) are engineering placeholders, not citations. The headline demo
  runs `ecc_off` with SEFIs off, so no reported number depends on them. They need a real multi-bit-
  upset fraction / per-transit probability before any `ecc_on` or SEFI number is quoted. **(Untouched
  in M4b — §4 stays open.)**
- **M4b is done on an NVIDIA L4 24GB** (not A100/H100): real-scale overhead, precision/recall, and
  calibrated-rate wall-clock are all re-measured above and labelled *NVIDIA L4 24GB*; the MPS/CPU
  tables below are kept and labelled separately. `DcgmXidSource` (real ECC/Xid polling) is
  implemented and validated on the L4. **Still open:** the demo video, the two citations above, and
  the ABFT-at-scale tolerance fix.

---

## Citations

All physics and tooling choices trace to `docs/research/technical-foundations.md`.

**Upset rates & environment.** COTS memory in LEO 1e-3…1e-7 upsets/bit-day; NASA NEPP design
criteria 1e-5…1e-7; deep-submicron flight data ~1e-9…1e-8 (Flying Laptop, 600 km SSO);
methodology CREME96. H100 80 GB HBM = 6.4e11 bits → 640–64,000 flips/day. SAA: 80–97% of LEO SEUs
occur in SAA transits (<20 min/orbit); Proba-II ~90%, a 1025 km satellite 97.3%; modeled as a
50–100× Poisson multiplier over ~10 min of a ~95-min orbit. Storm mode: Gannon storm, May 2024
(Wu 2025, *Space Weather*).

**Fault injection.** PyTorchFI (UIUC/NVIDIA) tensor-level approach, vendored/extended, not
hard-depended; NVBitFI (SASS-level) cited as a future validation path; tensor-level injection is
accepted for error-propagation studies (arXiv 2412.08466); NVIDIA beam vs simulation agree within
5× ("Demystifying GPU Reliability", 2021). SEFIs: Jetson RADECS 2024 — reboot cross-section
*exceeds* bit-error cross-section, so crashes matter as much as flips.

**Detection.** Real SDCs cause both loss spikes and silent convergence drift (AWS, "SDC in LLM
training", arXiv 2502.12340). ABFT overhead precedents: FT-CNN 4–8%; V-ABFT ~12% with variance-
based thresholds for fp16/bf16 (arXiv 2602.08043); ApproxABFT cuts exact-ABFT cost ~43%.

**Checkpoint / recovery.** PyTorch Distributed Checkpoint `async_save` (PyTorch ≥2.3);
CheckFreq ~3.5% overhead precedent (FAST'21); `cuda-checkpoint` + CRIU for transparent GPU
process snapshot/restore (CRIUgpu, arXiv 2502.16631, stretch).

**Positioning.** RedNet (arXiv 2407.11853) — per-layer selective protection for satellite DNN
*inference*, model-specific; we are a general runtime for unmodified PyTorch. KubeSpace
(arXiv 2601.21383) schedules containers across satellites — the layer above us. Google Suncatcher
(arXiv 2511.19468) proves COTS accelerators viable in orbit and explicitly leaves the software
fault-tolerance layer open — the wedge.

---

*Target: YC application demo. See `PLAN.md` for the full plan and `STATUS.md` for the build log.*
