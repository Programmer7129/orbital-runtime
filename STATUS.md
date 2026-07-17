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
