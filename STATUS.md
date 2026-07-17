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
