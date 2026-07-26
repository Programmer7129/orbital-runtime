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

## SUBMITTED ANSWERS — Fall 2026 form (name: Steadstar; SteadStar vs Steadstar TBD)

**STANDING RULE: use "I", never "we" (solo founder). Product = "the runtime"/"Steadstar"/"it".**

**Company name:** Steadstar
**Describe in 50 chars:** Radiation fault-tolerance for GPUs in orbit  (43 chars)

**What is your company going to make?**
Steadstar is a software runtime that keeps commercial GPUs computing through
radiation in orbit.

Datacenter GPUs are now being launched into space. The first NVIDIA H100 reached
orbit in November 2025, with Google and SpaceX following. Up there, cosmic radiation
strikes the chip mid-computation, and it fails in several distinct ways. A single
flipped bit can turn a weight into infinity and NaN the whole run. A multi-bit hit
slips past error-correcting memory. The chip hangs or reboots outright. And in the
worst case, the run finishes looking completely normal while quietly converging to a
worse model, with nothing to flag it. Error-correcting memory catches only some
single-bit flips; nothing protects the job itself, so today each operator hand-rolls
its own fix or goes without.

The runtime wraps an unmodified PyTorch job. It detects corruption in three tiers
(cheap guards on the loss and gradients, checksums on the matrix multiplies, and the
GPU's own error counters), catching the silent errors as well as the loud ones, then
recovers by rolling back to a checkpoint and replaying, whether the fault was a bad
number, a NaN, or a full crash.

It's already built and validated: on a rented datacenter GPU at true orbital
radiation rates, an unprotected training run dies while the identical protected run
completes, at ~1.6% detection overhead. Every constant is calibrated to NASA and
flight data. Next, I validate the fault model against a proton beam to measure how
each GPU generation actually fails, proprietary data that compounds per chip and per
customer.

**VOICE RULE:** human, candid, passionate, AND fully technical/measured. Never boil
down or drop supporting detail to sound casual. No drama ("paranoid", "the real
thing"). Include the knowledgeable insider tangents a true expert drops (UC Davis
Crocker / my alma mater / Google tested Suncatcher there / 67 MeV beamline). "I" not
"we". No "I commissioned"-style corporate tells.

**How far along are you?**
I've built the full runtime, end to end. It wraps an unmodified PyTorch job and does
three things: injects radiation faults calibrated to real orbital rates, detects the
corruption in three tiers (cheap guards on the loss and gradients, checksums on the
matrix multiplies, and the GPU's own error counters), and recovers by rolling back to
a checkpoint and replaying. There's a 288-test suite behind it and a demo anyone can
reproduce in one command.

The result that matters most: I rented an NVIDIA L4 datacenter GPU, turned the
radiation up to true orbital rates, and ran the same training job twice. The
unprotected run dies. The identical protected run finishes, and the detection costs
about 1.6% of runtime.

I've been careful with the physics, because a radiation simulator is only as good as
its calibration. Every constant traces back to NASA and real flight data. I ran two
hard adversarial reviews against the work, one on methodology and one on
reproducibility, and fixed everything they found. And I checked my fault model
against the proton-beam data Google published from irradiating its own space TPUs.

The next step is running that beam test for real. I'm planning it at UC Davis's
Crocker Nuclear Lab, which is my alma mater, and which is the exact 67 MeV proton
beamline Google used to qualify the Suncatcher TPUs. That campaign measures how each
specific GPU generation actually fails under radiation. Alongside it, I'm starting
design-partner conversations with the operators about to hit this problem, Sophia
Space, Axiom, and the ones coming after them.

**Are you looking for a cofounder?**
Yes. I've taken it this far solo, but I'm clear-eyed about the shape of team this
needs.

Physics isn't foreign to me, it's something I've loved for a long time. I earned a
silver medal in the International Astronomy and Astrophysics Competition, and I've
gone deep into the radiation-effects literature myself, that's how I calibrated the
fault model to NASA and flight data and checked it against Google's beam results. So
I can hold the physics, build the software around it, and tell good science from bad.

But there's a real difference between reading the field and being the person who has
spent years running proton-beam experiments and characterizing exactly how silicon
fails. That depth is what takes us from calibrated-against-published-data to
calibrated-against-our-own-beam-data, and it's what I want in a co-founder: a
radiation-effects scientist to own the science as Chief Scientist, from a world like
Vanderbilt's ISDE, NASA, or the UC Davis lab where I'm already planning the beam
campaign.

I'm searching actively now, including through YC's co-founder matching platform, and
reaching out to people in that field directly. I'd rather find the right complement
than stay solo by default.

*(Founder physics thread to reuse in founder video + "why this idea/domain
expertise": IAAC silver medal, self-taught the radiation-effects specifics, calibrated
the model personally. Not a physics tourist; just not the deep experimental
specialist, which is the co-founder gap.)*

**How long have you been working on this? How much full-time?**
About four weeks, and all of it full-time. I started at the beginning of July 2026
and have worked on it every day since, which is how it went from an idea to a
validated, GPU-tested MVP this quickly. I'm solo, and Steadstar is what I'm going
all-in on, 100% full-time from here.

**FRAMING RULE for competitors/moat (NEVER say "they can't build it"):** Any operator
can build a v1; several hand-roll one today. The claim is that they will BUY not
BUILD, via the cloud-consolidation dynamic. Radiation reliability is undifferentiated
infrastructure (no operator's edge is running its own bit-flip recovery, re-characterized
every chip generation). Early cloud looked identical: enterprises built private clouds
until the public providers' feature surface + economics made buying obvious. Steadstar's
surface compounds the same way: beam-validated per-chip failure data, cross-fleet
telemetry, flight heritage, then scheduling + assurance data on top. In-house v1 falls
behind; buying wins. Goal = be the layer they consolidate onto (AWS of orbital-compute
reliability), not the from-scratch build each abandons. This is AWS's own
"undifferentiated heavy lifting" framing. See interview-ammo.md "feature or company".

**Interview-only honest boundary (do NOT put on website/app, but say if asked
"what about latchup and total dose?"):** Steadstar handles single-event upsets,
multi-bit upsets, functional interrupts/crashes, and silent corruption. Latchup (SEL)
is a hardware short-circuit fixed by a power cycle; total ionizing dose (TID) is
lifetime wear. Both are hardware/shielding, not the software layer. Knowing exactly
where the layer ends is a strength signal.

## Remaining questions — to draft one by one
Cofounder question · founder video script · company URL · location · full-time
duration · tech stack + AI tools · why this idea/domain expertise/how do you know
need · competitors (NVIDIA argument done, cited) · make money/market size · other
ideas · why YC · users?(No) · revenue?(No) · fundraising?
