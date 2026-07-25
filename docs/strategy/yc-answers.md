# YC Application — working answers (Fall 2026)

Rule: every factual claim carries a citation to a primary/credible source before it
ships. Citations pending are marked [CITE].

## Q: Who writes code / non-founder work?

DRAFT (approved-pending-user-edits): Sole founder does all technical work; no human
non-founders, no contractor code, no prior-employer IP. Work method: founder architects
and directs AI coding agents (Claude-family) against written specs; independently
verified via two adversarial AI reviews committed to the repo (docs/reviews/), which
found real bugs subsequently fixed. Physics constants trace to NASA NEPP / CREME96 /
flight studies in-code. Deliverables: calibrated fault injector, 3-tier detection
(+1.6% overhead on NVIDIA L4), orbit-aware checkpoint/recovery, 270-test suite; L4
validation run: unprotected dies at step 179 at calibrated 1e-7 upsets/bit-day,
protected completes 300/300.

## Q: Competitors — "why won't NVIDIA just do this?" (the survival argument)

Five arguments, strongest first. EVIDENCE-BACKED versions required [CITE = agent
verifying now]:

1. NVIDIA ships primitives, not vertical reliability products — including on Earth.
   cuda-checkpoint is a primitive, not a product [CITE repo]. The terrestrial
   fault-tolerant-training layer was built by the ecosystem, not NVIDIA:
   ByteCheckpoint (ByteDance) [CITE], Meta's SDC detection program [CITE "Silent Data
   Corruptions at Scale"], CheckFreq/Gemini (academia) [CITE]. If they skipped the
   10,000x-bigger terrestrial market's reliability layer, "easily builds it for
   hundreds of orbital GPUs" inverts their operating history.
2. Attention, not resources, is NVIDIA's constraint: orbital GPU volume through ~2029
   is a rounding error against datacenter revenue [CITE volume estimates + NVIDIA DC
   revenue]. Their number doesn't move; the roadmap (next-gen architectures) does.
3. Incentive/liability misalignment: NVIDIA warranty/EULA does not cover spaceflight
   use [CITE exact warranty language]; an official "space mode" extends them into
   mission assurance, export-control-heavy sales, and radiation-performance claims
   they structurally avoid. Third parties can publish failure characterization the
   vendor won't.
4. The layer must be chip-neutral: operators fly mixed silicon (TPUs — Google
   Suncatcher; Jetson — Aethero/Sophia; H100-class — Starcloud). NVIDIA protects only
   NVIDIA. Precedent: Datadog vs CloudWatch — neutral layer wins because neutral.
5. If NVIDIA eventually cares, revealed behavior is BUY the ecosystem leader:
   Mellanox [CITE $6.9-7B, 2019-20], Bright Computing [CITE 2022], Run:ai [CITE
   ~$700M, 2024, GPU-orchestration SOFTWARE]. Plan = be the leader worth buying:
   cross-chip beam-validated fault models + fleet telemetry + flight heritage they
   cannot shortcut.

One-liner: "NVIDIA never built the fault-tolerance layer even for terrestrial
clusters — they ship primitives and buy ecosystem winners (Run:ai). Our plan is to be
the winner worth buying, holding the cross-chip beam data and flight heritage they
can't shortcut."

## Remaining questions — to draft one by one
Cofounder question · founder video script · company name + 50-char + "what will you
make" · location · progress/full-time · tech stack + AI tools · why this idea/domain
expertise/how do you know need · make money/market size · other ideas · why YC.
