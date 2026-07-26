# ICP value + GTM (research 2026-07-26) — how customers use us, why they pay

## The market, in one line
Every operator putting a COTS GPU in orbit is betting a chip built for a
climate-controlled datacenter will keep computing CORRECTLY through radiation, and
almost none can prove it before spending $30-60M to launch it. **Steadstar sells the
proof, then the protection.** The universal unsolved question (stated openly by Axiom,
Orbital, Starcloud; dodged by Madari, NTT) is: "does the COTS chip compute correctly
through radiation, and can we prove it before launch?"

## Status quo across operators (terse)
Sector pattern: almost everyone flies COTS GPUs (H100 / Jetson Orin / Thor / Vera
Rubin) + in-house software mitigation + shielding, because rad-hard silicon is ~1/100th
the FLOPS. Each is rebuilding the same checkpoint/scrub/ECC stack.
- **Starcloud** — flew first H100 (Nov 2025); in-house; whole 40MW/10yr economics rest
  on the chip computing correctly for a decade, unproven on a 1-chip demo.
- **Sophia Space** — building "fault-tolerant, radiation-aware" ODC software in-house;
  6-yr hardware refresh = degradation priced as recurring launch mass.
- **Axiom** — AxDCU-1 on ISS is literally a "does COTS survive?" test; paying launch $
  to answer what ground testing answers ~100x cheaper.
- **Aethero** — furthest along: COTS + shielding + hand-rolled checkpoint/scrub/ECC.
  Competitor-as-customer (buy a validated layer + sellable test data).
- **Orbital (a16z)** — 2027 Pathfinder's whole purpose is to test GPU radiation
  tolerance; pre-seed thesis rides on one de-risking mission.
- **Madari** — NO public radiation story at all; highest-risk, strongest pre-launch
  assessment prospect.
- **NTT/Space Compass, Loft, Little Place Labs, Blue Origin** — conservative/gov/
  multi-tenant buyers who must prove reliability to THEIR customers.

## The economics that make it worth paying (cited)
- Launch ≈ $3,000/kg customer (Falcon 9); Starcloud break-even ≈ $500/kg → every kg of
  redundant/shielding mass to cover radiation eats the margin that justifies the whole
  business.
- A node = $50-60M; ~40% of smallsat missions fail, clustering as early "infant
  mortality" from undertested parts (exactly what ground assessment catches).
- The compute is worth more than the satellite: frontier training run $50M-$1B+
  (Epoch); 2027 heading to $1-3B. In orbit you CAN'T hot-swap the bad GPU or cheaply
  re-run.
- KILLER LINK: radiation = silent data corruption. On the GROUND, SDC already forces
  12-43% checkpoint overhead = $120-430M waste on a ~$1B run (OCP/NVIDIA, LLM-PRISM).
  In orbit it's strictly worse (no hot-swap, no cheap re-run). That's the number that
  makes the software worth paying for.

## VALUE POINTS (6, each tied to a real pain)
1. De-risk the mission BEFORE the launch: ground sim tells them if their GPU+workload
   survives their orbit — the question Axiom/Orbital/Starcloud pay launch $ to answer.
2. Stop a bit-flip silently wrecking a $50M-$1B run you can't re-run in orbit (SDC
   protection, made critical because space removes cheap recovery).
3. Fly less redundant mass → direct margin (at $500-3000/kg vs $500/kg break-even).
4. Replace the hand-rolled stack → redeploy scarce eng onto their real differentiator.
5. Quantify radiation exposure for mission planning (orbit/inclination/shielding/refresh
   cadence) → turns Sophia's guessed "6-yr refresh" into a number.
6. Give them a reliability story for THEIR customers/procurement (NTT, gov, multi-tenant
   SLAs) — a third-party assessment + SLA-grade fault tolerance is a sellable artifact.

## WILLINGNESS TO PAY (insurance framing)
Cost of NOT having it: a wasted launch + node ($30-60M), or a corrupted training run
($50M-$1B+), or a mission-credibility failure that ends the company's contracts.
Price of the software: a rounding error against any of those. The pitch is NOT "buy
reliability software" — it's "you're about to bet $50M of launch + a nine-figure
workload on an untested assumption; for low-single-digit % of that, we prove the bet
or protect it." Sharpest for pre-revenue set (Orbital, Madari, Sophia): one bad demo
kills the next raise, so we de-risk the RAISE, not just the mission.

## PRICING (sequence matched to how they buy)
1. **NEAR-TERM WEDGE — Ground assessment / simulation engagement** (fixed-fee per
   campaign, ~$50-250k). Matches how space procurement buys de-risking (mission
   assurance / test campaigns); no on-orbit dependency; fast to sell; every operator
   needs it before every launch. START HERE.
2. **Per-GPU or per-satellite runtime license** (annual, scales with fleet) — on-orbit
   phase; revenue scales with FLOPS deployed.
3. **Subscription / platform** (exposure model + health monitoring) — sticky
   reliability-ops tooling for fleet operators.
4. **Usage / outcome-based** (per protected GPU-hour) — when workloads/SLAs are the
   sold product (multi-tenant hosts like Loft).

## THE VALUE SENTENCE
"You're about to launch a $50-million node to answer one question, will your GPU keep
computing correctly through radiation, and then bet nine-figure workloads on the
answer. Steadstar proves it on the ground before you fly and keeps the compute correct
once you do, for a fraction of a single wasted launch."

Cold-email tight: "Don't spend a launch finding out your GPU can't survive orbit, we
tell you on the ground and protect the run once it's up."

Full sources in the research-agent transcript; key: Epoch (training cost), OCP/NVIDIA
SDC, LLM-PRISM (arXiv 2604.10390), ScienceDirect (LEO SEU + smallsat failure), Falcon 9
cost/kg, Starcloud economics, per-operator links.

## BOTTOM-UP TAM (2026-07-26) — the model behind "how much could you make"
Method (accelerator "bottom-up TAM"): for each revenue stream, ICP x annual contract
value x count -> TAM; then capture % -> company size. Beats top-down "1% of a big
market" because it disciplines the unit price (see the pricing correction below).

**Stream A — pre-launch assessment (near-term wedge, 2027-2029)**
- ICP: operators flying COTS GPUs (~15 serious ones today).
- ACV: ~$150-250k/operator/yr (campaign-based; re-run per chip generation / per mission).
- Count: ~15 now -> ~40-50 by 2030 as new entrants launch.
- Near-term TAM: ~15 x ~$200k = ~$3M/yr. Small BY DESIGN — this is the land + the data
  harvest, not the business.

**Stream B — on-orbit runtime license (the business, 2030s)**
- ICP: same operators, once deployed. Unit: per-GPU-year (or per-satellite).
- Driver = GPUs in orbit. Anchors:
  - 1 GW ~= ~1M H100-equivalent GPUs (Starcloud).
  - ABI Research: ~1.5 GW effective / 18,600 datacenter satellites by 2035 -> ~1.5M GPUs
    (conservative, independent analyst).
  - Bull: Starcloud 5 GW design + 10 GW Crusoe deal + Anthropic/SpaceX "multiple GW"
    (May 2026) -> ~5-10M GPUs.
- Price: reliability license at ~5-10% of per-GPU compute spend.
- TAM:
  - Conservative (2035, 1.5M GPUs x ~$1.5k) = ~$2.25B/yr.
  - Bull (5M GPUs x ~$2k) = ~$10B/yr.

**PRICING DISCIPLINE (why bottom-up mattered):** implied compute spend = $39B market /
~1.5M GPUs = ~$26k per GPU/yr. So a reliability license is ~$1.5-3k/GPU/yr (a few %),
NOT the $10-30k/GPU/yr gut estimate — $10k would be ~40% of the compute bill, absurd.
Bottom-up caught an overprice a top-down never would. (Still a rounding error vs the
nine-figure workload it protects, but real money at fleet scale.)

**Market cross-check:** in-orbit datacenter market $1.78B (2029) -> $39.1B (2035),
67% CAGR (BIS/ResearchAndMarkets). Reliability layer = a few % of that = single-digit
$B/yr, consistent with the per-GPU build.

**Capture -> company size:** neutral standard + data moat -> winner-take-most ->
20-40% share.
- 25% of $2.25B = ~$560M/yr revenue (conservative).
- 25% of $10B = ~$2.5B/yr revenue (bull).
- Software margins -> multi-billion-dollar company either way.

Sources: ABI Research (space-datacenters forecast: 1.5 GW / 18,600 sats by 2035);
Starcloud (1M H100/GW, 5 GW plan, 88,000-sat path, Crusoe 10 GW deal); BIS Research /
ResearchAndMarkets (in-orbit DC market $1.8B->$39B, 67% CAGR); Anthropic-SpaceX
multi-GW interest (May 2026).
