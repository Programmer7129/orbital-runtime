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

## "Feature or company?" / "Anyone can rebuild your 2-day MVP" (THE question)

**Reframe the 2 days:** the 2-day artifact is the DEMO, not the product/moat. AI
compressed "build a credible demo" from months→days — for us AND every competitor. So
the demo was never the moat; AI commoditizing code is the ARGUMENT for why the moat
must be non-code: beam time, flight heritage, accumulated cross-chip failure data,
customer trust, first-mover platform position. We've been building exactly those.

**The cloud analogy works FOR us:** private-cloud→AWS proved undifferentiated infra
reliability is BOUGHT, not built. No operator's edge is "we hand-rolled bit-flip
recovery." Operators building in-house today = the "private cloud" phase; consolidation
onto a platform is the bet. (Also: AWS won not on EC2 alone but on the EXPANDING
surface — 200+ services raised switching cost until self-hosting made no sense.)

**Is there an expanding surface? YES — reliability is a privileged wedge** (sits
UNDER the customer's workload, on every satellite, sees everything — same launchpad
Datadog/CrowdStrike/Cloudflare expanded from a narrow start). Five expansion vectors:
1. Reliability → **observability/fleet-mgmt**: own the telemetry → health dashboards,
   predictive failure ("deorbit this sat before its HBM dies"). The Datadog path.
2. Reliability → **orchestration/scheduling**: we know which nodes are healthy / in the
   SAA / degraded — exactly what a scheduler needs. Beachhead to BECOME the
   workload-placement layer (the "layer above us," KubeSpace territory).
3. Reliability → **data/assurance business**: cross-chip beam-validated failure model
   is a saleable DATA ASSET — to insurers (pricing orbital-compute risk), chip buyers
   (which GPU survives orbit), mission designers. A second business the software emits.
4. Reliability → **cross-environment TAM**: same runtime → LEO, then lunar/deep-space
   (more radiation, more need), then TERRESTRIAL edge/HPC (Meta/Google SDC is proven).
   TAM = "any GPU where reliability is hard," not just orbital GPUs.
5. Reliability → **chip-generation treadmill**: every new chip (B200/Rubin/TPU v7)
   needs re-characterization → recurring, renews each generation. Subscription baked
   into the physics.

**Honest risk (say it):** the platform expansion is a THESIS, not a fact — today we
have one feature. Getting to expand requires winning the beachhead, the market
materializing, and out-executing whoever else wants the orchestration layer. The
"feature that gets acquired by NVIDIA / the orchestration winner" outcome is REAL and
FINE (Run:ai precedent) — not a failure mode. One-liner: "Resilience is the wedge, not
the whole company — it's a privileged wedge because it sits under everything and sees
everything, which is exactly how Datadog and CrowdStrike turned one feature into a
platform."

## Terrestrial bit flips: how Earth handles it + why it doesn't transfer to orbit

**How Earth deals with bit flips today (layered):**
- Layer 1 — ECC memory (NVIDIA-built, in silicon): silently corrects single-bit memory
  flips, flags double-bit; surfaced via Xid / DCGM. Catches the COMMON case → why most
  people never think about radiation. Guards MEMORY only, not the math.
- Layer 2 — SDC (silent data corruption in LOGIC/compute; ECC misses it). Discovered at
  fleet scale by Google ("Cores That Don't Count," HotOS'21) + Meta ("SDC at Scale,"
  2021): a few "mercurial cores" silently miscompute. Handled 100% IN-HOUSE via fleet
  screening + quarantine (find bad chip, remove it).
- Layer 3 — training resilience = checkpoint/rollback. Tools built by ByteDance
  (ByteCheckpoint), academia (CheckFreq FAST'21, Gemini SOSP'23) — NOT NVIDIA.
- WHO SELLS A NEUTRAL "keep-my-compute-correct" LAYER: nobody. Even the ecosystem layer
  is DIY, because only hyperscalers feel enough pain to build it.

**Q1 — Why can't Meta/ByteDance/Google do it for orbit too?**
They built a "handle our rare broken chips" tool, not a "radiation reliability" tool.
Different problem:
- Terrestrial SDC = PERMANENT manufacturing defects in a FEW chips → screen + quarantine
  + hot-swap. Orbital = TRANSIENT radiation strikes on EVERY healthy chip, intermittently,
  correlated to orbit position (SAA). Nothing to quarantine (chip is fine 10 min later),
  and you can't swap hardware in orbit. Their playbook ("find bad hardware, remove it")
  is meaningless in orbit.
- Their checkpoint cadence is tuned for a LOW rate; at the orbital rate it's uselessly
  sparse. The hard part is knowing WHICH strike is lethal — that's what Steadstar does.
- Data they lack: protecting vs radiation needs beam characterization of each chip. Meta
  has never put an H100 in a cyclotron. Not their expertise; their tools encode none of
  it. = the moat.
- Even if they solved it, they build for THEMSELVES. Meta isn't a satellite operator and
  won't become one to sell to Starcloud. Cloud-consolidation logic: they build in-house,
  the market buys from a neutral platform.
- One-liner: "Their tool finds the one broken chip in a million and removes it. In orbit,
  every chip breaks intermittently and none can be removed. Not a harder version of their
  problem — a different one, needing beam data they've never collected."

**Q2 — Why not serve terrestrial datacenters too?**
Could technically (same engine; Earth is the low-rate SUBSET), but leading there breaks
every Vrin-pivot lesson:
- Pain isn't acute on Earth: ECC + occasional checkpoint is "good enough" → low WTP.
- Only buyers who feel it are hyperscalers, who build in-house (Q1). So terrestrial =
  indifferent OR already-DIY'd. Crowded/free-ish — the market you learned to avoid.
- Moat is orbit-specific: beam per-chip failure data is worth most where radiation
  dominates; at sea level flux is ~100x+ lower and SDC is defect-driven, not
  radiation-driven → the unique asset is least differentiating on the ground.
- Focus: seed-stage wins by owning ONE acute, mandatory, unserved market. Orbit is that.
- BUT real expansion later: neutron soft errors matter terrestrially at scale/altitude
  (aviation, high-altitude DCs, huge fleets). Once you're the orbital standard with data
  moat + flight heritage, moving DOWN to terrestrial is natural. Orbit = wedge,
  terrestrial = later market.
- One-liner: "Same engine, and Earth is the easy subset. But on Earth it's a nice-to-have
  sold to a market that's indifferent or DIY. In orbit it's mandatory and unserved. I go
  where the pain forces a purchase, then expand down to terrestrial from strength."
