# Interview ammo — narratives, one-liners, and defenses (YC app + interviews)

## The LANL story (origin + why-they-stopped + why-that's-good)

**Origin:** Early 2000s, LANL's ASC Q supercomputer kept crashing; investigation traced
it to cosmic-ray neutrons flipping bits (Michalak et al.), aggravated by 7,300-ft
altitude (~5x sea-level neutron flux). Their answer: resilience SOFTWARE (ECC +
checkpoint/restart doctrine + fleet monitoring + beam-test-before-purchase), not
exotic hardware. The institution with computing's worst radiation problem concluded
software wins — 20 years before us.

**Why they went quiet (4 reasons, in order):**
1. They WON their version: mitigations hardened into ops checklists; solved problems
   stop being publishable. (Beam testing continues as a service; LAMP = up to $1B
   renewing LANSCE through 2050 — practice got a permanent budget, research moved on.)
2. The exascale-panic funding wave crested: machines arrived (Frontier, El Capitan),
   checkpoint+ECC sufficed, DOE anxiety budget moved to AI. FTXS workshop cancelled
   2025; reliability lead pivoted to AI-for-ops.
3. The problem emigrated: 2021-26 breakthroughs are Meta/Google fleet SDC and
   university GPU beam tests — commercial GPU fleets, not government CPU machines.
4. Their environment plateaued (5x, constant); orbit is 100-1000x, on chips getting
   worse per generation (H100 memory MTBE 3.2x worse than A100 — NCSA SC25,
   arXiv:2503.11901), running week-long training jobs with no technician on call.

**One-liner:** "LANL didn't abandon the problem — they tamed their 5x version into
standard practice and the funding moved on. Orbit is the 1000x version on worse chips
with no one on call. The field went quiet exactly when its biggest market was being
born — that's the opening, not a warning."

**The honest caveat + counter:** "Checkpoint+ECC was enough for them — so where's your
research?" → We're an ENGINEERING company: known techniques suffice in principle; what
doesn't exist is productization for orbit — calibrated to 1000x flux, beam-validated
for GPUs (training-under-beam is verified unmeasured in the literature), packaged so a
12-person satellite startup can adopt it. Kubernetes wasn't novel CS either; it was
the productization of what Google knew. That's us vs LANL.

## Other core narratives (see companion docs for citations)
- **Transmission vs computation disambiguation:** links have FEC (solved, decades
  mature — TBIRD 4.8TB error-free); our problem is bits corrupted INSIDE the GPU
  while computing. Mail-truck-with-tracking vs office-burglar-editing-documents.
- **NVIDIA survival argument:** 5 points, fully cited — docs/strategy/yc-answers.md.
- **Lineage:** Hamming'50 (ECC) → TMR/von Neumann'56 → Huang-Abraham'84 (ABFT) →
  HPC checkpointing → SpaceX commodity-chips+voting flight computers → Meta/Google
  SDC (2021) → us. Plus flight ancestor: LANL's Cibola Flight Experiment
  (SEU-recovering FPGA payload — flew).
- **Current operator "plans" = duct tape:** Google punted software layer (Suncatcher
  paper silent on it); Starcloud demo-grade in-house for 1 GPU; Orbital(a16z)
  architected around it (stateless inference only); Aethero bundles with own hardware;
  everyone else: spares. Pre-platform pattern (pre-Kubernetes, pre-Datadog).
- **Beam validation = distributions, not events:** weather-model analogy; NVIDIA's 5x
  agreement result is a cross-section comparison. Our campaign measures the unmeasured
  layer: training-run outcome distributions under beam (verified unclaimed,
  docs/research/beam-calibration-audit.md).
- **Rad-hard cost:** RAD750 (Perseverance rover, ~$200k) ≈ 1998 iMac-class CPU;
  ~10^6x compute gap vs H100. Rad-hard = million-x penalty; our runtime = ~2%.
- **Growing-problem stat:** every transistor shrink lowers critical charge → more
  upsets; H100 3.2x worse memory MTBE than A100; DDR5 ships on-die ECC because flips
  became a SEA-LEVEL problem.
