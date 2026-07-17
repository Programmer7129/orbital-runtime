# STATUS

Progress log for the builder session. One entry per milestone; newest last.

---

## 2026-07-16 — M0: scaffold + orbit model + Poisson engine — COMPLETE

**Audit of the prior partial scaffold**

| Artifact | Verdict |
|---|---|
| `orbital_runtime/orbit/track.py` | **Kept.** Well-cited, deterministic, phase-gated SAA. Two edits (see below). |
| `pyproject.toml`, `Makefile`, `README.md` | Kept as-is. |
| `orbital_runtime/orbit/__init__.py` | **Was broken** — imported `FluxModel` from a `flux.py` that did not exist, so the package would not import at all. Fixed by writing `flux.py`. |
| `tests/` | **Empty.** The scaffold claimed statistical tests; none existed. Written from scratch. |
| `demo/workloads/nanogpt/data/` | Empty directories only. M1 work. |
| venv | Healthy, but the package was **not** installed into it (`pip install -e` had never been run). Fixed. |

**What passed:** full suite green, 41 tests, ~9 s.
`tests/test_orbit_track.py` (geometry), `tests/test_flux.py` (calibration + statistics), `tests/test_rng.py` (determinism).

**Key numbers (measured)**

| Base rate (upsets/bit-day) | Upsets/day, H100-class 6.4e11 bits | Mean interval | SAA share |
|---|---|---|---|
| 1e-9 | 640.0 | 135.0 s | 89.8% |
| 1e-8 (default) | 6,400.0 | 13.5 s | 89.8% |
| 1e-7 | 64,000.0 | 1.4 s | 89.8% |

SAA share vs multiplier: 50x → 85.5%, **75x (default) → 89.8%**, 100x → 92.2%. All inside the
80–97% flight-data band; the default reproduces Proba-II's observed ~90%.
Daily total is invariant to the multiplier (see below) — 6,400/day at every M.

**Two real modeling bugs found and fixed (not test artifacts)**

1. **The SAA multiplier was inflating the cited daily total.** Published LEO rates
   ("1e-9 upsets/bit-day, Flying Laptop 600 km SSO") are *orbit-averaged* — observed
   upsets ÷ bits ÷ days, with SAA passes already inside the total. Multiplying the
   published rate by 75x inside the SAA manufactures upsets rather than redistributing
   them, and would have had the demo report ~5,600 flips/day at 1e-9 while citing a
   source that says 640 — an 8.79x overstatement that looks superficially plausible.
   Fixed by normalizing λ by the time-average multiplier `A = f·M + (1−f)`. Both
   research-doc anchors (640–64,000/day **and** 80–97% SAA share) now hold
   simultaneously; neither did before.
2. **`expected_upsets_per_day` under-reported by 0.9%.** It integrated λ(t) over a
   literal 86400 s = 15.16 orbits; the trailing 0.16 orbit contains no SAA transit
   (the window sits at phase 0.35–0.46), so the day captured 15 transits, not 15.16.
   Fixed by scaling one whole orbit to a day — the quantity flight data reports.

**Design decisions worth knowing**

- **`orbital_runtime/rng.py` (new).** Named, `SeedSequence`-derived independent RNG
  streams keyed by a BLAKE2b hash of the stream name (not `hash()`, which is
  PYTHONHASHSEED-salted and would break cross-process reproducibility — there is a
  subprocess test for this). Streams are addressed by *name*, not spawn order, so
  enabling a subsystem cannot perturb another's draws. This is what makes the
  protected-vs-unprotected demo comparison apples-to-apples: turning ABFT on must not
  shift the flip schedule.
- **Exact piecewise Poisson sampling**, not thinning: λ(t) is piecewise constant, so
  per segment `N ~ Poisson(λ·Δt)` with uniform placement. Statistical tests then check
  the model rather than a sampler's convergence.
- **`in_saa` now shares one window-edge definition with `saa_windows`** — the flux model
  builds segments from one and labels events with the other; they must not disagree.

**Blockers / honesty flags**

- ⚠️ **`DEFAULT_ECC_LEAK_FRACTION = 0.02` is an engineering assumption, NOT a cited
  number.** The research doc gives ECC-on behaviour qualitatively ("only multi-bit
  residuals + logic faults + SEFIs leak through") but no multi-bit-upset fraction, and
  I declined to invent one — PLAN.md acceptance criterion 3 requires every physics
  constant to trace to the research doc. Flagged in `flux.py`. **Any headline number in
  `ecc_on` mode needs a real MBU fraction first.** The default `ecc_off` mode (what the
  headline demo runs) is fully calibrated and unaffected.
- No environment blockers. Python 3.13.7, torch 2.13.0, MPS available, CPU/MPS only.

---

## 2026-07-16 — M1: break things — COMPLETE

**What passed:** full suite green, **138 tests, ~17 s**. New: `test_inject_memory.py` (31),
`test_inject_compute.py` (12), `test_inject_sefi_xid.py` (17), `test_failure_modes.py` (17),
`test_determinism.py` (11), `test_telemetry.py` (11).

**M1 deliverable met:** `orbital-run --protect off` reliably produces a corrupted run.
Across 8 seeds at the elevated rate, **8/8 runs are corrupted — 0 escape intact**:

| Outcome | Seeds | Meaning |
|---|---|---|
| Died (NaN) | 7/8 | failure mode (a) |
| Completed but degraded | 1/8 (seed 3) | failure mode (b), val 3.95 vs 2.99 clean (**+0.96**) |

`make demo` (0.81M-param model, 200 steps, 2 orbits, seed 1337): clean run reaches
val **2.4304**; identical irradiated run at 3e-5 **dies at step 39** after 49 delivered
upsets (42 in SAA). Deterministic.

**The finding that makes the story work: only fp32 bit 30 is catastrophic.**
For a typical weight (|v|<1) the biased exponent is ~120 = `01111000`, so exponent bits
27–29 are *already set* — flipping one **clears** it and drives the value to ~1e-12, which
is harmless (equivalent to zeroing a weight). Only **bit 30**, the one exponent bit that is
0, gets **set**, multiplying by 2^128. So ~**1 bit in 32 is lethal, not 4 in 32**. This is
why runs absorb ~100 upsets and then die from a single one — seeds 5 and 6 each died from
**exactly one flip**. Verified against `struct` independently of torch.

Consequence: the same uniform-over-bits injector produces both published failure modes with
no tuning. Which mode you get is decided by IEEE-754, not by a knob.

**Other findings worth carrying forward**

- **`flips_nonfinite` is 0 even in runs that die of NaN — this is correct, not a bug.**
  Bit 30 turns 0.01 into ~3e36: large but perfectly *finite*. The NaN is manufactured later,
  when that weight meets a matmul. Counting these as "non-finite flips" would misattribute
  the mechanism.
- **ReLU masking is real and reproduced.** A flip leaving a negative pre-activation negative
  is annihilated by the following ReLU — output bit-identical to clean. This is why an
  injected upset ≠ an SDC. **M2 consequence: recall must be measured against faults that
  PROPAGATE, not against faults injected**, or masked-and-harmless faults will be scored as
  detector misses.
- **Adam's second moment absorbs strikes.** A huge `exp_avg_sq` shrinks the update to ~0, so
  optimizer-state hits are often self-neutralizing. Part of why seed 3 survives 98 upsets.

**Three real bugs found by tests (all would have been silent)**

1. **`reshape(-1)` silently copies non-contiguous tensors** — a flip would have landed on a
   throwaway copy, producing perfect telemetry while the model trained on undamaged weights.
   The contiguity check ran *after* the reshape, so it never fired. Now checked on the
   original tensor, and it raises rather than no-oping.
2. **`static_resident_bits` over-counted 4/3.** AdamW keeps a 0-dim `step` scalar beside
   `exp_avg`/`exp_avg_sq`; counting state *tensors* instead of *bits* read AdamW as 3
   states/param and set λ **33% too high**. Now measured exactly when state exists.
3. **`strip_wall` missed `wall_s`** on `run_end`, which would have made every
   determinism-of-the-log check vacuous. Wall-clock fields are now enumerated in
   `WALL_CLOCK_FIELDS`.

**Design decisions**

- **One training loop for both runs.** `--protect off` is the same `train()` with detection
  and recovery disabled, not a separate path — otherwise measured overhead could be an
  artifact of the loop rather than of protection.
- **The whole fault timeline is drawn before step 0** from named streams, so protected and
  unprotected runs face a bit-identical bombardment (asserted in
  `test_flip_schedule_is_independent_of_protection`). This is what makes the headline
  comparison a controlled experiment.
- **Grad clipping left ON (nanoGPT's default 1.0).** It is a real baseline defense, so the
  runtime must show value *on top of* standard practice rather than against a strawman.
- **Batches keyed on `(seed, step)`, not a long-lived generator** — required for M3's
  exact replay after rollback.
- `--protect on` currently **exits 2 with an error** rather than silently running unprotected.

**Blockers / honesty flags**

- ⚠️ Elevated rates (3e-5 demo, 5e-4 tests) are **model-size compensation, not physics**: the
  demo model holds 7.8e7 resident bits vs an H100's 6.4e11. The calibrated 1e-9→1e-7 band is
  asserted against real H100 bit counts in `test_flux.py`. Documented in the Makefile and the
  test module. **M4 must produce headline numbers at realistic scale.**
- ⚠️ `DEFAULT_ECC_LEAK_FRACTION` (M0) and SEFI `p_per_transit` are both uncited assumptions.
  Both channels are **off by default**, so no current number depends on them.
- Xid stream is *correlated with* injected flips, not derived from simulated ECC hardware —
  M2 must report watcher recall as a plumbing measure, not a detector-accuracy result.

---

## 2026-07-16 — M2: see things — COMPLETE

**What passed:** full suite green, **208 tests, ~21 s**. New: `test_guards.py` (18),
`test_abft.py` (32), `test_watcher.py` (9), `test_detector.py` (15).

### Precision / recall vs known injected faults (12 seeds, rate 5e-4, CPU)

| Tiers | Precision | Recall | Median latency |
|---|---|---|---|
| tier 1 (guards) | 1.00 | 0.83 | 24 steps |
| tier 1+2 (guards+ABFT) | **1.00** | **1.00** | **2 steps** |
| tier 1+2+3 (+watcher) | 1.00 | 1.00 | 2 steps |

**False positives on 12 clean runs: 0/12.** Irradiated runs corrupted: 12/12 (10 died).
First detection came from ABFT in 9/12 and from `isfinite` in 3/12.

**ABFT's measured value: recall 0.83 → 1.00, and latency 24 → 2 steps (12× faster).**
Latency is not cosmetic — it is exactly the number of steps M3 must replay after rollback.
The watcher contributes nothing in `ecc_off`, which is correct and is the pitch: with ECC
off the hardware reports nothing, so application-layer detection is all there is.

**Ground truth is exact, not thresholded.** Determinism gives a free oracle: run the same
seed at rate 0 and irradiated, and the loss curves are bit-identical until the first fault
*propagates*. So `corruption_step = first step where losses differ` — no tolerance, no
judgement call, detector never consulted. This also settles M1's masking question: a fault
ReLU annihilates never moves the curve, so it is never scored as a miss. Recall is measured
against faults that propagate, which is the only kind that can hurt the model.

### Overhead per tier (measured; PLAN.md design rule 4)

| Config | CPU (0.81M) | MPS (0.81M) | MPS (10.7M) |
|---|---|---|---|
| A/A control = **noise floor** | 3.2% | 0.8% | 0.3% |
| tier 1 guards | below noise | below noise | below noise |
| tier 1+2 **adaptive** | **+7.2% ✓** | +22.3% ✗ | **+5.4% ✓** |
| tier 1+2 @100% sampling | +15.0% ✗ | +127.6% ✗ | +36.3% ✗ |

**Target <10% with ABFT sampling on: MET** at realistic scale (+5.4% on 10.7M params/MPS,
+7.2% on CPU). Tier 1 is genuinely free — below the noise floor everywhere, as claimed.

**100% sampling never meets the target (+15% to +128%). This is the quantitative case for
adaptive vigilance:** position-aware sampling covers the SAA — where ~90% of upsets land —
completely, at 19% average sampling, and that is the difference between missing the target
and meeting it.

**The benchmark was wrong before it was right (twice).**

1. **It reported NEGATIVE overhead (−4.3%) for tier 1** — impossible; watching two scalars
   cannot speed up training. Running all of A then all of B let thermal drift masquerade as
   signal. Fixed with round-robin interleaving **and an A/A control** — the baseline timed
   twice under different names, whose apparent overhead *is* the resolution limit. Effects
   below it are now reported as "below noise", never as a number. Without that control there
   is no way to distinguish a real 1% from a noisy 0%.
2. **ABFT was sync-bound, not compute-bound.** `.item()`/`argmax()` per check forced a
   GPU→CPU stall several times per step: **+203%** at full sampling on MPS. Now every value
   stays a device tensor and the whole step's checks resolve in **one** sync inside
   `observe()`. 203% → 128% full, 30% → 22% adaptive.

**Why overhead falls as the model grows (and what it means for M4).** ABFT's remaining cost
is kernel-launch overhead, not arithmetic: ~6 small kernels per sampled Linear. On a 0.81M
model each GEMM is microseconds, so the fixed cost dominates; at 10.7M the GEMMs are big
enough to amortize it (22.3% → 5.4%). The theoretical 1/out saving only materializes at
scale. **This is evidence M4's real-scale numbers should be better, not worse — but it must
be measured on the A100, not extrapolated.** (Profiled: `refresh_checksums()` is only 0.4 ms
of a 12.9 ms step — I had assumed it was the bottleneck and it was not.)

**Design decisions**

- **ABFT trusted-snapshot ordering is the whole tier.** Checksums are captured immediately
  after `optimizer.step()`, when weights are known-good, and the next forward is compared
  against that. Deriving `s` from the *live* weights instead — including the natural-looking
  `weight._version` cache, since an injected flip bumps the version — makes both sides of the
  identity contain the same bad weight: they agree, and every memory SEU in a Linear passes
  silently. `test_a_stale_or_live_checksum_would_miss_it` pins this so a refactor can't undo it.
- **Measured ABFT sensitivity floor: catches bits ≥15 (relative perturbation ≥3e-3), misses
  bits ≤12 (≤4e-4) — 17 of 32 bit positions.** Not a defect to tune away: below the floor the
  perturbation is indistinguishable from the GEMM's own rounding noise, and lowering the
  threshold just buys false positives on clean steps. Asserted in both directions.
- **Two kinds of signal, tracked separately.** `isfinite`/ABFT/fatal-Xid are proof
  (`Verdict.certain`); z-score/loss-spike are inference. M3 uses `certain` to decide whether
  to roll back immediately or escalate — rolling back on every blip would burn the budget.
- **Detectors are strictly read-only**, asserted on both clean and irradiated runs. If
  observing changed the trajectory, every number in `bench/` would be measuring a different
  computation than the one it claims.
- **`DcgmXidSource` raises rather than returning `[]`.** A watcher that reports "no errors"
  when it cannot see the device makes an unmonitored run look healthy — worse than no watcher.

**Bugs found**

- `strip_wall`-style NaN trap, again: `nan != nan` made loss-curve equality checks fail on
  any run that *died*, and would have made `first_divergence` report a bogus corruption at
  the NaN step. Both are now NaN-aware; equality elsewhere is still exact.

**Blockers / honesty flags**

- ⚠️ The +22.3% MPS small-model figure is real and reported. The target is met *at scale*,
  not universally. M4 must re-measure on the A100 rather than quote the 5.4%.
- ⚠️ Precision/recall is measured on the **tiny** test model (1 layer, 32-dim) at 5e-4 for
  suite speed. The mechanism is scale-free but the *numbers* are not — M4 should re-run
  `bench/detect_eval.py` at demo scale.
- Watcher recall in sim measures plumbing, not ECC physics (carried from M1, still true).

---

## 2026-07-16 — M3: survive things — COMPLETE

**What passed:** full suite green, **249 tests, ~30 s**. New: `test_ckpt.py` (23),
`test_recovery.py` (18, incl. CPU **and** MPS).

### The headline demo (`make demo`, seed 1337, 0.81M params, 2 orbits, 200 steps, MPS)

| Run | Outcome | Val loss |
|---|---|---|
| clean baseline (rate 0) | completed | **2.4304** |
| `--protect off` @ 3e-6 | **DIED (NaN) at step 141** | inf |
| `--protect on` @ 3e-6 | **COMPLETED 200/200** | **2.4288** |

The protected run absorbed 49 upsets, detected 7 (5 ABFT / 2 guard), rolled back 7 times,
replayed 105 steps, and **recovered a model matching the never-irradiated baseline to 0.07%**
(2.4288 vs 2.4304). Deterministic and reproducible.

**M3 acceptance:** protected runs survive what kills the unprotected one across seeds 1/2/5
(`test_protected_run_survives_what_kills_the_unprotected_one`); **bit-exact resume passes**
(`test_restore_is_bit_exact` — every parameter and optimizer tensor identical to the last
bit, plus `test_replay_after_restore_reproduces_the_original_trajectory`, which retraces the
original losses exactly).

**The honest limit — protection is not a magic wand.** At 10× the demo rate the protected run
still converts certain death into survival, but the model ends degraded:

| Rate | unprotected | protected val (clean = 2.4304) |
|---|---|---|
| 3e-6 | died @141 | **2.4288** — indistinguishable from clean |
| 1e-5 | died @43 | 3.3604 — survived, degraded |
| 3e-5 | died @38 | 3.3671 — survived, degraded |

The residue is exactly M2's measured detection floor: strikes on bits ≤12 perturb a weight
by <4e-4, sit below ABFT's noise floor, get baked into checkpoints, and accumulate. Rollback
cannot undo what nothing detected. At the calibrated flight band this is negligible; at 300×
that band it is visible, and it is reported rather than tuned away. **The demo rate is 3e-6
for this reason** (set `DEMO_RATE=3e-5` to see the degraded regime).

### The design decision that matters most: the clock counts EXECUTED steps

A rollback rewinds the training step. **The satellite does not fly backwards.** Keying sim
time to the training step would have rewound the orbit too — the run would re-enter the SAA
it just escaped, meet the identical scheduled upsets forever, and never progress. Worse,
**replay would be free**: the protected run could dodge radiation by rewinding the universe,
and the demo would measure a physical impossibility.

Keying to executed work makes replay cost what it should. The protected demo run executed 312
steps to train 200, flying **1.6× the nominal mission** and meeting 49 upsets where the
mission nominally scheduled 36. Verified by `test_replay_costs_orbit_time`. Consequence: the
schedule is drawn over a 3× horizon (`SCHEDULE_HEADROOM`), because a replaying run that flew
out of its own schedule would finish in an empty universe — silently flattering the exact run
we are trying to prove. `schedule_exhausted` reports it if that ever happens.

### M2's latency measurement became M3 behaviour

The rollback margin is **per detection reason**, not a global constant:

- **ABFT mismatch → margin 1.** The trusted checksum is exactly one step old, so a mismatch
  at step D localises the fault between D−1's update and D's forward. Any checkpoint ≤ D−1 is
  provably safe. Same for a driver-timestamped Xid.
- **NaN / z-score → margin 25.** These say only "something is wrong now"; M2 measured guard
  latencies up to 24 steps.

The first version used the pessimistic margin for everything. It threw away good checkpoints,
**replayed 37 steps where 1 would do, then ran out of eligible checkpoints and declared a
recoverable run unrecoverable.** Detection latency is not a slide statistic — it is literally
how much work a rollback burns.

**Bugs found (three, all real)**

1. **MPS crash in the checkpoint path — a design-rule-1 violation the tests missed.**
   `state_checksum` accumulates in float64, which MPS does not support. Every checkpoint test
   was CPU-only, so `make demo` (which defaults to `--device auto` → MPS) raised on the very
   machine this is developed on. Fixed by staging checkpoint tensors to CPU — which is also
   the more correct design, since they are bound for host storage anyway. `test_recovery.py`
   now runs the whole protected loop on every available device.
2. **A step-0 checkpoint could never be restored.** It predates Adam's state entirely, so a
   load template built from the live (now warm) state demanded `opt.*` keys that were never
   written, and DCP raised. Checkpoints now record exactly which keys they hold — and restore
   **drops** optimizer state the checkpoint predates, rather than leaving a warm, possibly
   corrupted moment attached to step-0 weights.
3. `state_checksum` over a NaN tensor returned NaN, so a corrupted checkpoint would compare
   unequal to *itself* — verification failing for the wrong reason. Non-finites now map to a
   deterministic sentinel: verification fails loudly and stably.

**Other design decisions**

- **Checkpoints are taken at the same trusted instant as ABFT's snapshot** — right after
  `optimizer.step()`, before radiation lands. Saving any later would persist the fault into
  the very checkpoint meant to escape it.
- **Double buffering earns its keep** because the *newest* checkpoint is the *most likely to
  be untrustworthy* (detection lags the fault). Every checkpoint self-verifies on restore; a
  failed verification falls back to the older slot instead of losing the run.
- **Best-effort rollbacks are counted separately** (`best_effort_rollbacks`). When nothing
  provably predates the corruption we try the oldest checkpoint anyway, but a recovery made
  on a guess must never be reported as a proven one — and it does not reset the failure
  streak that stops us looping on a bad checkpoint.
- **`RecoveryExhausted` is a real outcome**, surfaced as `death_reason="unrecoverable"`. A run
  that cannot recover gets reported, not retried forever.
- **`steps_completed` (progress) vs `steps_executed` (work)** are tracked separately, so a
  protected run can never take credit for redoing work it had already done.

**Blockers / honesty flags**

- ⚠️ Sub-detection-floor accumulation (above) is the real limitation. Not a bug — a measured
  boundary of what checksum-based detection can see.
- ⚠️ Recovery overhead is large at demo rates (105–195 replayed steps for 200 trained) because
  the rate is 300× the flight band. At calibrated rates rollbacks are rare. **M4 must report
  wall-clock overhead of `--protect on` at realistic rates**; the M2 overhead table measures
  detection only, not replay.
- All M0–M2 flags still stand (uncited ECC leak fraction and SEFI probability, both off by
  default; elevated rates as model-size compensation).

---

## 2026-07-16 — Handoff: exactly what M4 needs

M0–M3 are complete and committed; the suite is green at 249 tests. **Stopping here per
instructions.** M4 is "show things" — dashboard, cloud GPU, video, README.

### 1. Dashboard (`demo/dashboard/`) — nothing exists yet

The **only** input it needs is the JSONL telemetry at `runs/<tag>-s<seed>.jsonl`. It is
self-sufficient by design and asserted so (`test_run_log_tells_the_whole_story`). Read it with
`orbital_runtime.telemetry.read_events`. Events: `run_start`, `step`, `flip`, `activation`,
`sefi`, `xid`, `detect`, `checkpoint`, `rollback`, `run_end`. Every record carries
`seq`/`kind`/`step`/`t_sim`/`wall`.

- Orbit map + SAA shading: `OrbitTrack.ground_track(t_sim)` gives (lat, lon);
  `track.in_saa(t_sim)` gives the shading window. **The ground track is display-only** — SAA
  membership is phase-gated, not lat/lon-derived (see `track.py`).
- Dual loss curves: `step` events from the two runs' logs.
- Upset counters: `flip` events carry `in_saa`, `bit`, `value_before/after`, `target_kind`.
- Overhead ticker: `run_end` carries `wall_s`, `steps_executed`, `replayed_steps`.
- ⚠️ **`train.py` emits `step` events only every `log_every` (default 10).** A smooth
  dashboard curve needs `TrainConfig(log_every=1)` or interpolation.
- ⚠️ A protected run's log contains **repeated step numbers** (replay). That is truthful, not
  a bug — the dashboard should show the rollback as the story, not dedupe it away.

### 2. Cloud GPU (rent A100/H100) — the numbers that must be re-measured there

Nothing in the repo is CUDA-specific; `resolve_device("auto")` picks CUDA when present. Re-run:

- `python -m bench.overhead --device cuda --n-layer 12 --n-embd 768` — **do not quote the
  MPS/CPU figures.** Expect *better* than +5.4%: ABFT is kernel-launch-bound, and the cost
  amortizes as GEMMs grow (measured 22.3% → 5.4% going 0.81M → 10.7M). Evidence, not proof.
- `python -m bench.detect_eval --device cuda` at demo scale — the current precision/recall
  (1.00/1.00) is measured on a **1-layer, 32-dim** model at 5e-4.
- **Protected-run wall-clock overhead at calibrated rates (1e-9..1e-7)** — never yet measured.
  The M2 table is detection cost only and excludes replay.
- At H100 scale (6.4e11 bits) the **calibrated** rates finally bite: 640–64,000 flips/day
  without any elevation. The M4 demo should not need an elevated rate at all — that alone is
  a strong slide.

### 3. Real DCGM/Xid (M4 is when PLAN.md permits it)

`detect/watcher.py` already has the interface. Implement `DcgmXidSource.poll()`; it currently
raises rather than returning `[]` deliberately. `WatcherTier` needs no change.

### 4. Citations still owed (blocking acceptance criterion 3)

- **`DEFAULT_ECC_LEAK_FRACTION = 0.02`** (`flux.py`) — a placeholder, not a citation. Needs a
  real multi-bit-upset fraction before any `ecc_on` number is quoted. `ecc_off` (the demo
  default) is unaffected.
- **SEFI `p_per_transit`** (`sefi.py`) — the RADECS finding is qualitative; no per-transit
  probability. Channel is off by default.
- Both are flagged in-code. Everything else traces to `docs/research/technical-foundations.md`.

### 5. README results table

Numbers ready to quote (all reproducible, all in this file): the demo table above, M2's
precision/recall and overhead tables, M1's 8/8-corrupted result, and M0's calibration anchors
(640–64,000 flips/day; 89.8% SAA share vs the 80–97% flight band).

### Suggested M4 order

1. `log_every=1` + dashboard against existing JSONL (no GPU needed; highest demo value).
2. Rent the GPU, re-measure the four items in §2, then record the video.
3. Chase the two citations in §4 — or state them as assumptions on the slide. Do not quietly
   ship the placeholder.

---

## 2026-07-17 — M4a: dashboard + demo script + README (the no-GPU items) — COMPLETE

**Scope was M4 PART 1 only** (no GPU, no video, no touching the two uncited constants), per
the builder brief. M4b (everything needing a rented GPU) is spelled out at the bottom.

**What passed:** full suite green, **262 tests, ~32 s**. New: `test_dashboard_build.py` (6).

**Audit of the WIP checkpoint (`1591a7b`) — kept in full, it was correct.** It added (a) a
JSON-safe encoding for non-finite floats — a dead run's `run_end` carries `final_loss: NaN`
and `final_val_loss: Infinity`, which `json.dumps` spells as bare `NaN`/`Infinity` that
`JSON.parse` rejects, so the one record explaining the death was exactly the record a browser
could not read; now written as the strings `"NaN"`/`"Infinity"`/`"-Infinity"` and decoded back
by `read_events`, with `allow_nan=False` turning any unencoded escapee into a loud failure at
the write — and (b) the `--log-every` CLI flag the handoff called for. Both are load-bearing for
the dashboard; nothing was reverted.

### 1. Dashboard (`demo/dashboard/`)

- **`build.py`** reduces the three telemetry logs to one bundle and writes `telemetry_data.js`
  (`window.TELEMETRY = {...}`). The page loads it via `<script src>`, **not** `fetch` — a
  `file://` page can't fetch a sibling file (unique-origin), so the demo needs **no server, no
  CDN, nothing but the two files**. Orbit display geometry is precomputed here from the **real
  `OrbitTrack`** (ground track + phase-gated SAA), so the page cannot drift into a second,
  contradictory copy of the physics.
- **`index.html`** — self-contained (inline CSS/JS). Orbit **ring** with the SAA as a shaded
  **phase arc** (0.35–0.455), not a lat/lon polygon — this is the honest rendering, since SAA
  membership *is* a phase window and the ground track is display-only (labelled as such on the
  page). Live counters (upsets / detected→rolled-back / steps replayed / ABFT coverage), dual
  loss curves (protected vs unprotected vs clean baseline) with a **"you are here" marker that
  jumps backward on each rollback** (the recovery shown, not deduped), a scrolling recovery
  event log, status tiles, and a wall-clock overhead ticker + final banner.
- **Colours are dataviz-validated:** protected `#0891b2` / unprotected `#d97706` pass all six
  checks on the dark surface (CVD ΔE 19–27); every curve is also direct-labelled, so identity
  never rests on colour alone.
- **Verified by actually rendering it** in Chrome from T+00:00 through completion: death burst
  at step 141, protected completing 200/200, all counters landing on the telemetry's numbers
  (49 / 7 / 105 / val 2.4288).

### 2. `demo/run_demo.sh`

Trains the three identical seeded runs (`baseline` rate 0 · `unprotected` 3e-6 dies ·
`protected` 3e-6 survives) with `--log-every 1`, bundles the dashboard, opens it. **Deterministic
end to end** — re-running produces a **byte-identical `telemetry_data.js`** (no live timestamp on
purpose). **~25 s wall on this Mac** (MPS), far under the 5-min target. Tolerates the unprotected
run's non-zero death exit; warns loudly if it ever fails to die. `NO_OPEN=1` skips auto-open.

`telemetry_data.js` is committed (the dashboard works on a fresh clone with no build step);
`runs/` stays git-ignored.

### 3. `README.md`

Rewritten from the stub: results tables (demo, precision/recall, per-tier overhead, calibration),
architecture, run instructions, honesty flags, and full citations traced to the research doc.

**Honesty (PLAN rules held).** The 3e-6 demo rate (≈300× the flight band, model-size
compensation) is disclosed in the dashboard, the README, and `run_demo.sh`. The overhead ticker
shows **wall-clock incl. replay** (+117% at the demo rate) and says so; the **calibrated
detection-only** figure (+5.4% at scale) is carried separately in the README so the two are never
conflated. The two uncited constants (`DEFAULT_ECC_LEAK_FRACTION`, SEFI `p_per_transit`) were
**not touched** and remain off by default.

---

## What M4b still needs (unchanged from the handoff §2–§4 above — all GPU-gated)

The no-GPU work is done; everything left requires the rented A100/H100 and was explicitly out of
this session's scope:

1. **Real-scale numbers on CUDA** (handoff §2): re-run `bench/overhead.py` and `bench/detect_eval.py`
   at demo scale (`--n-layer 12 --n-embd 768`), and measure **protected-run wall-clock overhead at
   calibrated rates (1e-9..1e-7)** — never yet measured; the ticker's +117% is a demo-rate artifact.
   At H100 scale (6.4e11 bits) the calibrated band delivers 640–64,000 flips/day **with no
   elevation**, so the M4b demo should drop the elevated rate entirely — the strongest possible slide.
   **Do not quote any MPS/CPU figure as the headline.**
2. **The video** (handoff §2): record the dashboard running the real-scale story once the numbers
   above exist. The dashboard is built and deterministic; it only needs the GPU-scale JSONL. When
   re-pointing it, note the dashboard currently hard-expects `runs/{baseline,unprotected,protected}-s<seed>.jsonl`
   and a `--protect off` rate-0 baseline; a calibrated-rate run may not `--protect off`-die within
   200 steps, so the demo may need more steps or a longer mission (the dashboard scales to `t_max`
   and `steps_requested` automatically — no front-end change needed).
3. **Real DCGM/Xid** (handoff §3): implement `DcgmXidSource.poll()`; interface already in place.
4. **The two citations** (handoff §4): a real multi-bit-upset fraction and a SEFI per-transit
   probability, or state them as assumptions on the slide. Still off by default; nothing shipped
   depends on them.

Nothing in M4a blocks any of the above. **Stopping here per instructions.**
