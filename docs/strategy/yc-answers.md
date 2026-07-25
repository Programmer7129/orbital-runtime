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

Five arguments, strongest first — ALL CITATIONS VERIFIED 2026-07-24 (primary sources;
two corrections applied during verification, noted inline):

1. NVIDIA ships primitives, not vertical reliability products — including on Earth.
   cuda-checkpoint is an explicit utility-with-limitations, not a supported product
   (github.com/NVIDIA/cuda-checkpoint). The terrestrial fault-tolerant-training layer
   was built by the ecosystem, not NVIDIA: ByteCheckpoint (ByteDance, Apache-2.0,
   arXiv:2407.20143, NSDI'25); Meta's SDC program ("Silent Data Corruptions at Scale,"
   arXiv:2102.11245; "Detecting SDCs in the wild," arXiv:2203.08989); Google's "Cores
   that don't count" (HotOS'21, doi:10.1145/3458336.3465297); CheckFreq (USENIX
   FAST'21); GEMINI (SOSP'23, doi:10.1145/3600006.3613145). If they skipped the
   vastly larger terrestrial market's reliability layer, "easily builds it for
   hundreds of orbital GPUs" inverts their operating history.
2. Attention, not resources, is NVIDIA's constraint: Data Center revenue was $75.2B
   in ONE QUARTER (+92% YoY, Q1 FY2027 reported May 20 2026; CNBC) — ~$35M/hour.
   Every plausible orbital GPU deployment through 2029 is a rounding error against
   the number their best people are paid to move.
3. Incentive/liability misalignment — EXACT QUOTE from NVIDIA's own product
   documentation (CUDA Installation Guide, Notices, docs.nvidia.com): "NVIDIA
   products are not designed, authorized, or warranted to be suitable for use in
   medical, military, aircraft, SPACE, or life support equipment..." (emphasis ours).
   Enterprise SLA §8.13 additionally disclaims all liability for "Critical
   Applications" (NVIDIA-Software-License-Agreement-2026.5.7). PRECISION NOTE: the
   license text says "Critical Application," not "space" — only the hardware notice
   names space explicitly; never overstate this in the interview. An official "space
   mode" would extend NVIDIA into mission assurance and radiation-performance claims
   they structurally avoid; a third party can publish the failure characterization
   the vendor won't.
4. The layer must be chip-neutral: operators fly mixed silicon (TPUs — Google
   Suncatcher w/ Planet, blog.google Nov 4 2025; Jetson — Aethero/Sophia;
   H100-class — Starcloud-1, first DC-class GPU in orbit Nov 2 2025, per NVIDIA's own
   blog: blogs.nvidia.com/blog/starcloud). NVIDIA protects only NVIDIA. Precedent:
   Datadog vs CloudWatch — the neutral layer wins because it is neutral.
5. If NVIDIA eventually cares, revealed behavior is BUY the ecosystem leader:
   Mellanox ($6.9B announced Mar 2019, closed Apr 2020 at ~$7B; NVIDIA Newsroom);
   Bright Computing (Jan 2022, HPCwire); Run:ai (GPU-orchestration SOFTWARE, ~$700M
   reported, announced APR 2024, closed DEC 30 2024 — corrected dates; SiliconANGLE).
   Plan = be the leader worth buying: cross-chip beam-validated fault models + fleet
   telemetry + flight heritage they cannot shortcut.

One-liner: "NVIDIA never built the fault-tolerance layer even for terrestrial
clusters — they ship primitives and buy ecosystem winners (Run:ai). Our plan is to be
the winner worth buying, holding the cross-chip beam data and flight heritage they
can't shortcut."

## Remaining questions — to draft one by one
Cofounder question · founder video script · company name + 50-char + "what will you
make" · location · progress/full-time · tech stack + AI tools · why this idea/domain
expertise/how do you know need · make money/market size · other ideas · why YC.
