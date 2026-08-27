# orbital-runtime

**A software runtime that keeps commercial GPUs computing correctly through radiation in orbit.**

Datacenter GPUs are going to orbit by the thousands (Starcloud, Google Suncatcher, Axiom, SpaceX
filings). Cosmic radiation corrupts their computation. Error-correcting memory catches part of it.
Nothing protects the job itself.

This runtime wraps an **unmodified PyTorch job**. It injects faults at published orbital rates,
detects the corruption that results, **repairs most of it in place**, and rolls back only when it
cannot. Every physics constant traces to a citation.

![Steadstar architecture](docs/assets/architecture.svg)

---

## The idea that makes it work

Detection is not the hard part. **Response** is.

A rollback throws away the model, reloads a checkpoint, and redoes every step since — while
radiation keeps arriving. At 85 million parameters that cost 76 replayed steps and the run still
died.

So the runtime does not roll back for a single corrupted number. It **finds that number and undoes
it**. Two checksums over the same tensor are enough: the plain sum says something changed, and the
index-weighted sum divided by it says exactly where.

![Repair instead of rollback](docs/assets/repair.svg)

Measured on an NVIDIA L4 at 85.3M parameters under 49 radiation hits: **39 of 42 corruptions
repaired exactly, 3 needed a rollback, the run completed.**

---

## Headline results

Measured on a rented **NVIDIA L4 24GB**, **85.3M-param GPT-2-class nanoGPT**, 8.19e9 resident bits,
300 steps, one orbit, at the **calibrated 1e-7 upsets/bit-day flight rate with no elevation.**

| | Unprotected | Protected |
|---|---|---|
| Outcome | **DIED — nan_loss at step 73** | **COMPLETED 300/300** |
| Upsets delivered | 2 | **49** (43 in the SAA) |
| Corruptions repaired in place | — | **39 of 42** |
| Rollbacks | — | 3 |
| Steps replayed | — | 25 |

Both runs face a **bit-identical bombardment** drawn before step 0 from named RNG streams, so
protection cannot change the radiation. It is a controlled experiment, not a demo.

### Detector recall, by measured GPU fault class

Recall is reported per class, because a blended number hides the class that dominates real faults.

| Path / class | Detection | Escalation | Missed |
|---|---|---|---|
| memory / nullification | **100%** | 82.5% | 0 |
| memory / bit flip | **100%** | 11.7% | 0 |
| memory / special (NaN) | **100%** | 96.2% | 0 |
| compute / nullification | 97.5% | 97.5% | 3 |
| compute / bit flip | **39.2%** | 39.2% | 73 |
| compute / special (NaN) | **100%** | 100% | 0 |

**Memory path: 100% share-weighted. Compute path: 69.3%.**

Detection and escalation are separate columns on purpose. A fault that is repaired or deliberately
absorbed was **seen**. Counting it as a miss would be dishonest in the flattering direction.

---

## Gate scoreboard

Thresholds were set **before** measurement, and two of them fail.

| Gate | Target | Result | |
|---|---|---|---|
| Fault model matches published data | ±1.5pp | **±0.22pp** | ✅ |
| Coverage of strikeable state | ≥90% | **100%** (was 29.8%) | ✅ |
| Memory-path recall | ≥70% | **100%** | ✅ |
| Headline: protected run survives | required | **300/300** | ✅ |
| Compute-path recall | ≥70% | **69.3%** | ❌ |
| Protection overhead | <10% | **25.7%** | ❌ |

The 70% floor is NVIDIA's own EUD diagnostic recall in ByteDance production (SOSP '25). The point of
publishing a failing scoreboard is that a passing one you cannot check is worth nothing.

---

## What changed, and why

Four defects were found by measuring rather than assuming. Each would have been found by a hostile
reviewer instead.

**1. Two thirds of the target surface had no detector.** ABFT verifies `nn.Linear` weights. Adam
keeps `exp_avg` and `exp_avg_sq` per parameter, so optimizer state is **66.7% of every strikeable
bit** — and it was unwatched. Worse, a flip there flowed into the weights at the next
`optimizer.step()`, and `refresh_checksums()` then recorded that corrupted weight as trusted. The
fault laundered itself into ground truth.

A new **integrity tier** (`detect/integrity.py`) closes it with exact integer checksums. Summing the
integer view is bitwise exact, so there is no tolerance for a fault to hide under.

**2. The fault model simulated the wrong failure.** The injector produced only memory bit flips.
Tung et al. (NVIDIA, DSN 2026) measured 600 million real GPU corruptions: **50.68% are
nullification** — a value zeroed — which a bit-flip model cannot produce at all. The injector was
tested against the minority case.

`inject/gpu_model.py` now reproduces the published distribution, split by path:

| Outcome | Ours | Published | Δ |
|---|---|---|---|
| Nullification | 50.90% | 50.68% | +0.22pp |
| Bit flip | 48.17% | 48.31% | −0.14pp |
| NaN / ±INF | 0.93% | 1.01% | −0.08pp |

`bench/fault_model_audit.py` exits non-zero if this drifts. It is CI-able.

**3. The old model was ~25× too lethal.** Bit positions were drawn uniformly, putting ~25% of flips
in the fp32 exponent. The measured NaN share is 1.01%. Correcting it dropped per-event lethality by
about 25×, and several earlier claims rested on the inflated number.

**4. Perfect detection without a response policy was harmful.** The exact checksum caught a bit-2
flip in an Adam second moment — a 5e-7 relative change — and paid a full rollback for it. Radiation
lands during the replay, so it rolled back again. Dead at step 112 of 300, from five negligible
upsets.

Fixed three ways: repair-in-place removes most rollbacks, a severity policy absorbs what physics
says will decay, and the checkpoint history went from 2 slots to 4 (two consecutive rollbacks used
to exhaust it entirely — that was the true cause of "unrecoverable").

---

## The overhead problem, and the plan

Protection costs **25.7%**, against a target under 10%. That is the worst-performing gate. The cost
is not the checksums themselves — it is how many times the runtime walks the data.

![Overhead optimization plan](docs/assets/overhead-plan.svg)

Three measured findings:

1. The weighted checksum costs **3× the plain one**, because it builds an index list as large as the
   data itself — 80 MB allocated and discarded, four times per step.
2. The runtime makes **4 full passes** over ~1 GB of state per step. Two are unavoidable (a before
   and an after). Two exist only because the plain and weighted sums are computed separately.
3. There are **304 separate GPU calls** per pass, at roughly 22 µs of launch overhead each.

The fix removes the index list rather than optimizing it. Laid out as a grid instead of a line, a
position is just a row and a column, so the index arrays shrink to the square root of the data —
**9,200 labels instead of 85,000,000, a 1,581× reduction.** Row and column sums then yield *both*
checksums from one structured pass, and the plain sum is free (`plain = sum(row sums)`).

| Change | Effect |
|---|---|
| Grid decomposition | **1.56× faster**, measured |
| One pass yields both sums | 4 passes → 2 |
| Batch the small tensors | 304 calls → a handful |
| **Projected** | **25.7% → 8–12%** |

Status: **researched and measured, not yet implemented.**

---

## Running it yourself

```bash
make venv install     # .venv + editable install + dev deps
make test             # full suite, ~40 s
demo/run_demo.sh      # three runs + dashboard (NO_OPEN=1 to skip auto-open)

# the audits that make the numbers checkable
python bench/coverage_audit.py       # what the detectors can structurally see
python bench/fault_model_audit.py    # injector vs published GPU distribution; non-zero exit on drift
python bench/recall_by_class.py      # what they actually catch, per fault class

# sweep the knobs
.venv/bin/python -m orbital_runtime.run --rate 1e-7 --protect on --orbits 1 --steps 300
```

The dashboard (`demo/dashboard/index.html`) is self-contained local HTML and JS. No server, no CDN.
Rebuild it from existing logs without retraining: `python demo/dashboard/build.py`.

---

## Architecture

```
orbital_runtime/
├── orbit/     track.py  — parametric 95-min LEO orbit; phase-gated SAA window (~10 min/orbit)
│              flux.py   — time-varying Poisson intensity: base × bits × SAA × storm
├── inject/    gpu_model.py — measured GPU outcome distribution (Tung et al. 2026), split by path
│              memory.py    — stored-state faults: MBU clusters, write-path nullification
│              compute.py   — activation faults: nullification, warp-aligned tracks, logic tiles
│              sefi.py, xid.py — SEFIs (Suncatcher-calibrated), Xid stream
├── detect/    guards.py    — tier 1: isfinite + grad-norm z-score + loss spike (≈free)
│              integrity.py — tier 2: EXACT checksums over all resident state; locate-and-repair
│              abft.py      — tier 3: sampled checksums around nn.Linear GEMMs
│              watcher.py   — tier 4: ECC/Xid consumer (real nvidia-smi ECC on the L4)
├── ckpt/      saver.py, policy.py, recover.py — DCP checkpoints, 4 slots; orbit-aware cadence
└── run.py, telemetry.py — CLI and the JSONL event log the dashboard reads
bench/  coverage_audit.py · fault_model_audit.py · recall_by_class.py · overhead.py
tests/  316 passing
```

**Two ideas do the work.**

*Adaptive vigilance* — detection intensity and checkpoint cadence keyed to orbital position. About
90% of upsets arrive in about 10% of the orbit, so a uniform budget wastes most of itself. The
narrow, defensible novelty is position-aware protection scheduling for general-purpose GPU
*training* runtimes. Radiation-aware *instrument* safing is decades-old spacecraft practice.

*Locate and repair* — classical algorithm-based fault tolerance applied to stored state rather than
to a matrix product. The OCP consortium white paper *Silent Data Corruption in AI* (2025, authored
across NVIDIA, Meta, Google, AMD, Intel, ARM and Microsoft) names ABFT integrated into core AI
kernels as the promising path to forward error recovery, and asks publicly for standardized SDC
resilience benchmarks. That is what `bench/fault_model_audit.py` is.

### Design rules (enforced, not aspirational)

1. **Device-agnostic core** — runs on CPU/MPS; CUDA-only paths sit behind interfaces with
   simulation fallbacks.
2. **Calibration is sacred** — every physics constant carries a comment citing its source.
3. **Rate and outcome stay separate** — outcome is a property of the silicon and transfers from
   terrestrial measurement. Rate is a property of the environment and stays orbital. Mixing them is
   the error that produced 597 zeroed elements per event in a first draft of the fault model.
4. **Determinism** — the fault schedule is drawn before step 0 from named RNG streams, so protected
   and unprotected runs face identical bombardment.
5. **Overhead honesty** — measured against an A/A control. Effects below the noise floor are
   reported as "below noise", never as a number.

---

## What is not yet proven

- **Compute-path recall is 69.3%, under the 70% floor.** `compute/bitflip` sits at 39.2% because
  those activation corruptions fall under V-ABFT's 2.2e-5 relative tolerance — the price of zero
  false positives. Tightening it trades false positives back in. This is the genuine research
  problem, and it is what a beam campaign and a radiation-effects co-founder would address.
- **Overhead is 25.7% against a <10% target.** The plan above is measured but unimplemented.
- **The injector is not beam-validated.** It is calibrated against published beam and flight data
  (Suncatcher, MICRO'21, NSREC'21) and now against NVIDIA's measured GPU outcome distribution. That
  is calibration, not validation. Validation is a proton beam campaign at UC Davis Crocker on the
  same 67 MeV beamline Google used for Suncatcher.
- **No public radiation SDC rate exists for any modern datacenter GPU.** The newest NVIDIA
  datacenter part with published beam data is the V100 (2017), and those FIT values are normalized
  "to not reveal business-sensitive information". The absence is a disclosure choice, not a research
  gap — which is precisely why running the campaign is worth something.
- **Tensor-level injection, not SASS-level.** "Demystifying GPU Reliability" found beam and
  simulation agree within ~5×, but that validation is for instruction-level fault models. It is
  cited as evidence that fault-injection simulation *in general* tracks beam data, not as a claim
  that this injector is beam-validated.

---

## Citations

All physics and tooling choices trace to `docs/research/technical-foundations.md`.

**GPU fault outcomes.** Tung, Huang, Saxena, Shirvani, Hukerikar, Jain, Gongalore (NVIDIA) and
Tyagi, "The Anatomy of Silent Data Corruption: GPU Error Pattern Study and Modeling Guidance",
arXiv 2605.04213, DSN 2026 — 600M corruptions across 25,000 SDC cases, 3M+ simulator hours.
Nullification 50.68%, non-special bit flips 48.31%, NaN/±INF 1.01%; single-bit flips under 40% of
GPU bit-flip events against 72–98% in CPU studies; warp-aligned periodicity at 2/4/8/16-element
spacing; control-logic faults corrupting 20–75% of streaming-multiprocessor output.

**Upset rates and environment.** COTS memory in LEO 1e-3…1e-7 upsets/bit-day; NASA NEPP design
criteria 1e-5…1e-7; deep-submicron flight data ~1e-9…1e-8 (Flying Laptop, 600 km SSO); CREME96
methodology. SAA carries 80–97% of LEO SEUs in under 20 min per orbit; modeled as a 50–100× Poisson
multiplier. Storm mode: Gannon storm, May 2024 (Wu 2025, *Space Weather*).

**Orbital beam data.** Google Project Suncatcher, arXiv 2511.19468 — the first published radiation
test of a modern datacenter AI accelerator. UC Davis Crocker Nuclear Laboratory, 67 MeV protons,
v6e Trillium TPU. SDC at 14.4–20 rad/event (σ 6–9e-9 cm²/chip), SEFI at 1 per 5 krad (σ ~2e-11), no
TID hard failures to 15 krad(Si). "Core logic and on-chip SRAM were the most SEE-sensitive
components, primarily manifesting as Silent Data Corruption." Their own open question: "the impact
of SEEs on training jobs, and the efficacy of system-level mitigations, requires further study."

**Fault injection.** PyTorchFI (UIUC/NVIDIA) tensor-level approach, vendored and extended.
Multi-bit upsets: MICRO'21 (doi 10.1145/3466752.3480111; V100 HBM2, ChipIR) — 31.5% of memory upset
events multi-bit, ~75% byte-contiguous. ECC as SDC→DUE redistribution: NSREC'21 (arXiv 2108.00554)
— ECC-on cuts SDC up to 21× but raises DUE up to 13.7×. ECC does not solve this: neutron beam data
on NVIDIA GPUs shows "ECC fails at reducing the occurrence of Critical SDCs", with the critical-SDC
share rising from 8% ECC-off to 61% ECC-on.

**Detection and the field.** Real SDCs cause both loss spikes and silent convergence drift — AWS
and Harvard, "Understanding Silent Data Corruption in LLM Training", arXiv 2502.12340: "although
the pretraining loss remains similar, SDCs can cause model parameters to drift away from
ground-truth weights". Meta: "the corrupted values are exchanged as true values, causing the
training to appear to progress without actual improvement." Google Gemini 1.0: "we can expect SDC
events to impact training every week or two." OCP white paper *Silent Data Corruption in AI*
(August 2025, NVIDIA + Meta + Google + AMD + Intel + ARM + Microsoft) endorses ABFT in core AI
kernels and calls for standardized resilience benchmarks. ABFT overhead precedents: FT-CNN 4–8%;
V-ABFT ~12%; ATTNChecker (PPoPP 2025) for attention.

**Checkpoint and recovery.** PyTorch Distributed Checkpoint `async_save`; CheckFreq ~3.5% overhead
precedent (FAST'21).

**Positioning.** RedNet (arXiv 2407.11853) protects satellite DNN *inference* per-layer and is
model-specific; this is a general runtime for unmodified PyTorch training. NVIDIA NVSentinel is
open-source GPU fault remediation that explicitly does no correctness checking — it quarantines bad
hardware and never asks whether completed work was right. Suncatcher proves COTS accelerators are
viable in orbit and leaves the software fault-tolerance layer open.

---

*See `PLAN.md` for the plan and `STATUS.md` for the build log.*
