# Founder study plan — field fluency in 1.5 days

Depth levels: **L1** = one-line fluency (can name it, place it, move on).
**L2** = mechanism fluency (can explain HOW to a smart layperson, whiteboard-level).
**L3** = ownership (can defend numbers, derive results, survive expert cross-exam).
Rule: our software stack and its physics interface are L3; everything upstream decays
toward L1 with distance from the product.

## CRITICAL CORRECTION FIRST (a live example of depth calibration)
The CMB (Cosmic Microwave Background) has NOTHING to do with bit flips — it's relic
photons from the early universe, microwave-energy, utterly harmless. The actual
culprits: (1) **galactic cosmic rays** — protons/heavy nuclei accelerated by
supernova shockwaves; (2) **solar particle events**; (3) **trapped protons** in the
Van Allen belts — the South Atlantic Anomaly is where the inner belt dips lowest
(Earth's magnetic dipole is offset/tilted); (4) at ground level, cosmic rays smash
air molecules → **neutron showers** (LANL's problem); (5) historically,
**alpha particles from chip packaging** (Intel's famous 1978 DRAM bug). Saying "CMB"
in an interview = credibility torpedo. This is why depth calibration matters.

## DAY 1 MORNING — Layer A+B: the particles and the flip (~3h)
**A. Space radiation environment — L2** (skip: cosmology, CMB, particle zoo)
- GCRs, solar events, Van Allen belts, why the SAA exists (offset dipole → belt dips
  to ~200km over South Atlantic), atmospheric neutron showers, packaging alphas.
- Self-check: explain to a friend why a satellite gets hit 75x harder for 10 minutes
  per orbit, and why Denver computers see more errors than Miami's.
**B. How a strike flips a bit — L2, taxonomy L3** (skip: TCAD, LET spectra math)
- Charge deposition in a junction; **critical charge (Qcrit)**; why shrinking
  transistors → smaller Qcrit → MORE upsets per device generation (your instinct is
  right and it's our growth thesis).
- The taxonomy — know cold, these are our product's vocabulary: **SEU** (bit flip),
  **MBU** (multi-bit, 31.5% of HBM events), **SET** (logic glitch), **SEL** (latchup,
  destructive), **SEFI** (chip-level crash/hang — dominant on ECC-on GPUs!), **TID**
  (cumulative dose wear — Suncatcher's 2 krad HBM threshold), **SDC vs DUE**
  (silent wrong answer vs detected crash).
- ECC basics: SECDED, why single-bit correct + double-bit detect; interleaving.

## DAY 1 AFTERNOON — Layer C+D: measurement and consequences (~4h)
**C. How the field measures — L2, cross-section L3** (needed for Crocker/faculty calls)
- **Cross-section (cm²)**: effective target area; upsets = sigma x fluence. Read our
  beam-calibration-audit.md and be able to explain every row's units.
- Beam testing: why 1M-x accelerated flux works (Poisson linearity), protons (Crocker,
  matches LEO trapped protons) vs neutrons (LANSCE, matches atmosphere/ground).
- Rate models: CREME96 / AP8-AP9 exist and what they take/give — L1-L2 only.
**D. What upsets do to ML workloads — L3, OUR DOMAIN**
- fp32 anatomy: sign/exponent/mantissa. DERIVE the bit-30 result yourself on paper:
  for |v|<1 the exponent's high bits are set except bit 30; flipping set bits →
  value→~0 (masked); setting bit 30 → x2^128 → NaN cascade. ~1 lethal bit in 32.
- Masking: ReLU annihilation, Adam second-moment absorption — why most flips don't
  matter and why that makes detection scoring subtle (propagated-fault oracle).
- Training vs inference vulnerability: inference = stateless, restart; training =
  weeks of accumulated state; optimizer state as silent-corruption reservoir.
- ECC-on residual threat model: MBU leakage + SEFI + logic — why "just use ECC" is
  wrong (NSREC'21: ECC-on shifts failures to crashes, DUE > SDC by 2.2-2.7x).

## DAY 2 MORNING — Layer E: our technology — L3 ABSOLUTE (~4h, hands-on)
- **ABFT by hand:** take a 3x3 matrix, append row/column checksums, multiply, verify
  a corrupted product cell is caught. 20 minutes, permanent understanding of
  Huang-Abraham 1984. Then: why fp16 rounding noise makes thresholds hard (V-ABFT).
- **Checkpoint/restart mechanics:** enumerate what must be saved for bit-exact resume
  (weights, optimizer, RNG streams, step counter, data position) — and why our
  hostile review's resume-off-by-one bug mattered.
- **Run the repo:** `make test`, `demo/run_demo.sh`, open the dashboard, then read
  README end-to-end, then BOTH hostile reviews (docs/reviews/) and STATUS.md honesty
  flags. Every disclosed limitation = an interview question you can now pre-empt.
- **Adaptive vigilance:** why position-aware sampling gives SAA coverage at 19%
  average cost; why checkpoint-before-SAA is free insurance.

## DAY 2 AFTERNOON — Layer F: history + landscape + self-test (~3h)
- Re-read: docs/strategy/venture-notes.md (market map), interview-ammo.md
  (narratives), yc-answers.md (NVIDIA argument), lanl-landscape.md, technical-brief.md.
- Timeline to internalize: Hamming'50 → alpha-particle bug '78 → Huang-Abraham'84 →
  Belgian election '03 → LANL ASC Q '05 → Cibola flies '07 → SpaceX voting computers →
  Meta/Google SDC '21 → Starcloud H100 '25 → Suncatcher '25 → us.
**SELF-TEST — answer cold, out loud, no notes:**
1. Walk me from a supernova to a NaN in a loss curve. (A→B→D chain)
2. Why doesn't ECC solve this? 3. Why can't NVIDIA just ship this?
4. What did Google measure at Crocker and what did they NOT measure?
5. How do you validate a random simulator against a random beam?
6. Why did LANL go quiet and why isn't that bad for you?
7. What's your overhead and why should I believe your benchmark?
8. What breaks your model? (SEFI-off history, MBU clustering, fp32-only, SAA
   idealization — you VOLUNTEER these; disclosed weakness = credibility)
9. Whose work are you standing on? (lineage, by name and year)
10. What would kill this company? (honest: NVIDIA moving early, orbital compute
    stalling, beam data refuting the simulator — and the mitigation for each)

## EXPLICIT SKIP LIST (tangents that burn your 2 days)
CMB/cosmology · particle physics beyond "charged particles deposit charge" ·
semiconductor band theory · rad-hard CIRCUIT design (SOI/TMR silicon — L1 one-liner
only: "$200k, 1998 performance") · optical-link physics/FEC (one disambiguation line
only) · orbital mechanics beyond "90-min orbit, SAA 10 min, sun-sync exists" ·
CREME96 internals · TCAD/Monte-Carlo device sim.
