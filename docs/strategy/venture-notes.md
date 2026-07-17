# Venture strategy notes (consolidated from planning session "orbital-yc", 2026-07-16)

Durable record of ~15 research agents' findings + decisions. Companion to PLAN.md
(build) and docs/research/technical-foundations.md (physics/tooling citations).

## 1. The company

**Product:** chip-agnostic radiation fault-tolerance runtime for commercial GPUs in
orbit — fault detection (guards + sampled ABFT), orbit-aware checkpointing, invisible
job recovery. Sold to orbital-compute operators.

**One-line pitch:** "Commercial GPUs are going to orbit by the tens of thousands and
cosmic rays crash them constantly. We're the software that makes a $30k H100 act like a
$30M rad-hard computer — and we're the only ones selling it."

**Why now:** Starcloud (YC S24 → $1.1B in 17 months, YC's fastest unicorn) proved
in-orbit GPU compute; Google Suncatcher 2027 prototypes; SpaceX ~1M-satellite orbital-AI
FCC filing; Blue Origin TeraWave Q4 2027. YC Summer 2026 RFS — authored by Starcloud's
CEO — explicitly requests space compute "optimized for mass, thermal performance, and
radiation."

**Whitespace (verified July 2026):** no independent vendor sells this layer. Starcloud
built demo-grade FT in-house for 1 GPU; Aethero bundles software with THEIR hardware;
Klepsydra targets ESA rad-hard processors not COTS GPUs; Red Hat/Crusoe/Antaris own
container orchestration (the layer ABOVE); RedNet (arXiv 2407.11853) is model-specific
inference hardening, no code. GitHub has zero maintained "radiation-tolerant PyTorch
runtime" repos.

## 2. The moat (answer to "why won't they build in-house")

1. **Beam-test data:** per-chip empirical failure characterization requires scarce beam
   time (Crocker/TAMU/LBNL, booked months out, $10-50k/campaign) and expires every chip
   generation. Vendor amortizes across customers; each operator alone pays full freight.
2. **Talent:** rad-effects ∩ CUDA/PyTorch expertise is a near-empty set (NSREC/RADECS
   guild ~ hundreds of people, mostly at NASA/primes). 20 operators can't each hire it.
3. **Cross-fleet telemetry + flight heritage:** every customer satellite feeds real
   upset data back (CrowdStrike model); "flight-proven on N missions" is
   quasi-certification; future insurance/assurance hook.
- **Honest weaknesses:** top of market (SpaceX/Google) always in-houses; NVIDIA "space
  mode" is the kill risk (mitigate: chip-agnostic, own competitor-chip data, speed);
  core techniques are published — the software alone is NOT the moat.

## 3. Market map (July 2026, all facts verified by research agents)

- **Operators:** Starcloud ($170M A, Benchmark, 88k-sat filing; Starcloud-2 late 2026
  w/ Crusoe as cloud layer); Orbital/a16z ($5M, Euwyn Poon, stateless-inference thesis,
  Pathfinder 2027); Sophia Space ($13.5M, Leon Alkalai ex-JPL Fellow + Rob DeMillo,
  TILE modules w/ Jetson Orins, demo on Apex bus late 2027); Axiom ODC (nodes launched
  Jan 2026 on Kepler sats, Red Hat Device Edge stack); Madari Space (UAE, PoC Q3 2026);
  Lonestar (lunar storage/DR).
- **Compute boxes:** Aethero ($8.4M Kindred; Jetson; Titan mission Oct 2026 = 16k TFLOPS
  distributed K8s on EnduroSat FRAME-15); Ramon.Space, Zero-Error Systems (rad-hard HW);
  Ubotica/Unibap/Spiral Blue (EO edge).
- **Ops software:** Antaris ($28M A, ex-Planet COO/CTO, SatOS); Constellation Space
  (YC W26, ConstellationOS fleet AI-SRE); Loft Orbital ($326M+, Cockpit/Ultimate Edge —
  channel or competitor, not customer); Little Place Labs (Orbitfy, SBIR-funded grind).
- **Ground/relay:** Cascade Space (YC Sp25, Crew Dragon comms architect); Apolink
  (YC S24, solo 19yo, $140M LOIs); Observable Space ($90M + $94M USSF via merger with
  PlaneWave); Kepler (ESA HydRON €18.6M prime).
- **Who pays today:** operators buy ~zero startup software (build in-house or Red Hat/
  Palantir/Crusoe). First revenue = government (DIU HSA pilot 2026, SDA HALO, AFWERX
  TACFI/STRATFI, APFIT; SBIR lapsed Oct 2025). Realistic yr 1-2: $0.5-3M, ~80% gov.
  Commercial opens 2027+ when fleets exist.

## 4. Funding-pattern lessons (from founder research on 14 companies)

Every YC space acceptance pulled ≥2 of: (1) "my last job WAS this hard problem"
specificity; (2) LOIs/design partners as revenue substitute ($140M LOIs got a solo
teenager in); (3) explicit "this just became possible" timing wedge. Outside YC:
flown hardware > pedigree > gov money; software-only founders got funded by engineering
the thesis around their weakness (Poon), merging with heritage (Observable), or grant
grind (Little Place — the cautionary tale). Constellation Space (W26) proves juniors
with brand names + working AI demo clear the bar.

## 5. UC Davis / advisor plan

- **Eric Prebys** — Director, Crocker Nuclear Lab (the exact 67 MeV proton beamline
  Google used for Suncatcher TPU tests). Access advisor. ~0.25-0.5%.
- **Jason Lowe-Power** — UCD CS, chairs gem5 (standard fault-injection simulator), NSF
  CAREER 2025. Technical advisor, approach FIRST with the demo. ~0.5%, growable.
- **Stephen Robinson** — 4x shuttle astronaut, directs UCD Center for Spaceflight
  Research; campus CubeSat pipeline (REALOP-1) = possible free flight demo. ~0.25-0.5%.
- Advisor equity market: 0.1-1% w/ 2yr vest + cliff (FAST agreement). 10% = cofounder
  tier ONLY (reserved for a working CSO from Vanderbilt ISDE / NASA Goddard / Aerospace
  Corp — the SEE device-physics gap UCD cannot fill).
- **IP rule for academic collaboration:** company software is built entirely outside
  the lab; collaboration scope = "experimental characterization only" (they measure, we
  build); authorship ≠ ownership; paper through UCD InnovationAccess standard agreement.
  First campaign: validate injector vs proton beam on Jetson Orin (~$5k HW; beam
  $10-30k if fee-for-service; $0 for the LOI/scheduling conversation — that sentence is
  all YC needs). NVIDIA precedent: beam vs simulation agree within ~5x.
- **Publish the bit-30 finding** (arXiv): credibility + rad-guild flypaper + O-1A/EB-1A
  evidence for Vedant.

## 6. ICP / Mom Test (paused for usage limits; partial findings saved)

Targets: Sophia (DeMillo/Alkalai), Aethero (Ge/Pinnamaneni — semi-competitor), Little
Place Labs, Orbital (Poon), Madari, Starcloud (probe in-house at 100x scale), Axiom/
Kepler/Skyloom eco; secondary: SkyServe, D-Orbit, Loft, EnduroSat, Apex, Planet.
Key probe (tests the in-house question empirically): "What did you do about SEU handling
on your last mission? What did it cost? How confident at 100x scale?" — never "would
you buy?". Conferences: IEEE SMC-IT/SCC Pasadena Aug 3-7 2026; SmallSat Salt Lake City
Aug 23-26 2026.

## 7. Build status (see STATUS.md for live state)

M0 ✓ orbit/SAA Poisson engine (calibrated 1e-9..1e-7 upsets/bit-day, 89.8% SAA share).
M1 ✓ injector + nanoGPT victim (8/8 seeds corrupt: 7 NaN, 1 silent 3.95-vs-2.99; bit-30
discovery: ~1-in-32 bits lethal, runs absorb ~100 flips then die of one; 3 silent bugs
self-caught incl. non-contiguous-reshape flips hitting a throwaway copy).
M2 ✓ detection (P=1.00 R=1.00, 2-step latency, 0 FP; adaptive ABFT +5.4% @ 10.7M params
MPS; SAA-adaptive sampling = full coverage @ 19% average sampling).
M3 ✓ checkpoint/recovery (committed 586662b; verify pending).
M4 pending: dashboard + rented A100/H100 headline numbers + demo video (needs user's
cloud account). Known uncited constants (quarantined from demo path): ECC leak fraction
0.02, SEFI p_per_transit — get real numbers from a rad physicist or state as assumptions.
Builder = background Claude session "orbital-builder" 6920588f (Opus 4.8), attach via
`claude attach 6920588f`; watchdog monitor in planner session.
