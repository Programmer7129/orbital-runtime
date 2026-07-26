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
and have worked on it every day since. That solo, full-time month is how it went from
an idea to a validated, GPU-tested MVP this fast. I'm now bringing on a co-founder to
run the science, and Steadstar is where I'm 100% committed from here.

**Who writes code / was any done by a non-founder?**
All of it is me. The system design, the physics model, the runtime, the tests, the
GPU validation, that's all my work, and no other person has written any of it. No
contractors, no employees, no co-founder yet, and no code carried over from anywhere
else.

I should be direct about how I build: I use AI heavily. I direct Claude models as
coding agents (Fable 5 orchestrating, Opus 4.8 building) against specs I write, and I
verify everything myself, including two adversarial AI reviews I ran against the
codebase that caught real bugs I fixed. So the code is AI-implemented but
founder-designed, directed, and verified. The only human on this product is me.

**Tech stack (incl. AI models/tools)?**
The product is Python and PyTorch. The runtime wraps an unmodified PyTorch training
or inference job: a custom Poisson bit-flip injector calibrated to orbital rates,
three-tier detection (loss and gradient guards, checksums on the matrix multiplies,
and the GPU's own hardware error counters), and checkpoint/recovery built on PyTorch
Distributed Checkpoint. There's a 288-test pytest suite, and I validate on rented
NVIDIA GPUs on AWS, an L4 for the real-radiation runs. The thesis site is Next.js and
Tailwind with KaTeX for the math.

The system design is entirely mine: the architecture, the physics model, and how
injection, detection, and recovery fit together. I ideated and designed all of it
before it was built. Where I move fast is execution, through an unusually deep AI
workflow in Claude Code. Claude Fable 5 runs as the harness and orchestrator, holding
the plan and dispatching the work; Opus 4.8 is the workforce doing the heavy
building; and I distribute development across parallel agent teams. I write the specs,
the agents implement against them, and I verify everything myself, including two
adversarial AI reviews that caught real bugs I then fixed. That setup is how one
person went from an idea to a GPU-validated MVP in four weeks. The thinking, the
system design, the physics, and the verification are mine; the AI is the execution
engine.

**Are people using your product?** No.

**When will you have a version people can use?**
The core already runs, anyone with a GPU can install it and reproduce the demo today,
so the software itself isn't the blocker. What's left is getting it into an operator's
hands, and because the runtime wraps an unmodified PyTorch job, integration is light.
I can have a design partner running it against their own workload in the next few
weeks.

The proton-beam campaign runs in parallel, not as a gate. It's a validation and
learning experiment: it confirms the fault model against real beam data and produces
per-chip failure data no one else has, which sharpens the product and builds the moat.
But nothing about it holds up a design partner starting now.

**How do or will you make money? How much could you make?**
I make money two ways, and they stack.

First, before an operator launches, I sell a ground assessment. They run their actual
model and GPU through my radiation simulator, backed by my own proton-beam data, and get
a report on whether their workload survives their target orbit and what protection
costs. It lands the design partner and gives me their real workload data. The beam work
is a one-time R&D cost per chip generation, not something I re-run per customer, so each
assessment is cheap to deliver.

Second, once they fly, Steadstar becomes the runtime on the satellite, protecting the
live workload, licensed per GPU or per satellite with a monitoring subscription on top.
That is the recurring business, and it scales with the compute they deploy.

Here is the math, bottom-up.

The inputs:
- 15 operators: roughly the entire serious field flying commercial GPUs today,
  concentrated enough that I can reach all of them directly.
- $200k per operator per year for assessments: this maps to how space teams already buy
  mission-assurance and pre-launch test campaigns.
- 1.5M GPUs in orbit by 2035: ABI Research forecasts about 1.5 GW of orbital compute by
  then, and Starcloud puts roughly 1M H100-equivalent GPUs in a gigawatt.
- $2k per GPU per year for the runtime license: each GPU's compute is worth about $26k a
  year (the $39B orbital-datacenter market divided across 1.5M GPUs), so this is under a
  tenth of the value it protects.
- 20% capture: deliberately conservative, the share I would hold as the neutral standard
  operators consolidate onto.

The calculation:
- Assessments today: 15 x $200k = $3M a year. Small on purpose, this is the wedge.
- Runtime market by 2035: 1.5M x $2k = $3B a year.
- My share: 20% x $3B = $600M a year, at software margins.

So my best estimate is a $600M-a-year company on a conservative 20% slice of a market
that has barely started, and multiples of that if the gigawatt-scale plans (Starcloud's
5 GW, its 10 GW Crusoe deal, Anthropic's multi-GW interest) land.

(Full ICP value/pricing research: docs/strategy/icp-value-gtm.md. Bottom-up TAM model +
sources: same file, "BOTTOM-UP TAM" section.)

**Why this idea? Domain expertise? How do you know people need it?**
Why I picked it: it's where my two obsessions meet. I'm an AI-infrastructure engineer
by profession, and physics is what I've chased on my own time for years. I earned a
Silver honour in the International Astronomy and Astrophysics Competition, and I
studied quantum-computer hardware and quantum error correction through TU Delft's
courses, where I first got pulled into the problem of protecting fragile computation
from physical errors. So when I watched everyone race to put commercial GPUs in orbit,
the part that grabbed me wasn't the hardware or the launch, it was that nobody had
solved keeping the computation correct through radiation. That's fundamentally a
physics problem solved in software, my exact intersection.

Domain expertise: I'm a CS graduate from UC Davis with a real physics foundation, a
working AI-infrastructure engineer, and I've taken research-stage ideas to production
before, so I know how to turn something hard into something customers can use. I
calibrated this fault model to NASA and flight data and validated it on a real GPU
myself. I'm honest that I'm not the deep experimental radiation-effects specialist yet,
that's the co-founder I'm recruiting, but I have more than enough physics and systems
depth to build this, judge that scientist's work, and lead the company.

How I know people need it: they're already doing it the hard way. Google beam-tested
its TPUs and published the data. Starcloud flew an H100 and hand-rolled its own
mitigation. Axiom's ISS node exists to answer "does COTS silicon survive orbit?",
they're spending launch dollars on the question I answer on the ground. Orbital's
entire 2027 mission is to test GPU radiation tolerance. Sophia is building
fault-tolerant software in-house; Madari has no radiation story at all. Every serious
operator is either rebuilding this themselves or paying to learn it the expensive way.
That's demonstrated need, not a guess. And it follows the cloud pattern. Before AWS,
every company ran its own servers, because there was no alternative, and engineers
spent most of their time on infrastructure that had nothing to do with their actual
product. Amazon named that "undifferentiated heavy lifting" and built a platform to
take it off their hands, and the market consolidated onto it. Radiation reliability is
the undifferentiated heavy lifting of orbital compute: every operator is rebuilding the
same fault-tolerance stack instead of working on their satellites and their models. I'm
building the platform they outsource it to.

**Who are your competitors? What do you understand that they don't?**
There's no vendor-neutral company doing exactly this yet, so the real competition is
three things, plus one theoretical.

First, and the one I take most seriously: in-house teams. Starcloud, Sophia, and
Aethero each hand-roll their own fault-tolerance stack. That's the build-it-yourself
competitor.

Second, Aethero, which bundles radiation mitigation with its own Jetson hardware, but
only protects Aethero's boxes, not the H100s and TPUs everyone else flies.

Third, the status quo: rad-hardened silicon, which "competes" by avoiding the problem
at orders of magnitude higher cost and a fraction of the performance. Operators are
escaping it, not adopting it.

The theoretical one is NVIDIA. But NVIDIA's entire playbook is to build the foundational
platform and let the ecosystem build on top of it. They created CUDA, then let others
build the frameworks and the layers above it, PyTorch came from Meta, TensorFlow from
Google, the fault-tolerant-training tooling from ByteDance, Meta, and academia. NVIDIA
makes the silicon and the low-level platform; it has never built the application and
reliability layers that run on top. Radiation reliability for orbital GPUs is exactly
one of those layers. They'll make the chips that fly; I'm building the layer that keeps
them computing in orbit.

What I understand that they don't: this isn't a hardware problem or a one-time software
fix, it's a data problem. What actually protects a workload is knowing exactly how each
chip fails under radiation, and that data only comes from beam characterization,
expires every chip generation, and compounds across every customer's chips and fleets.
An operator building in-house gets one satellite's worth and redoes it each generation;
a neutral platform accumulates it across the whole market. That's the undifferentiated
heavy lifting of orbital compute, and it consolidates onto a platform the way cloud
did. The operators building it themselves are in the pre-AWS phase of this market.
Every GPU that flies will need this layer, and I'm building it to be the platform the
whole industry runs on.

**RULES (investor-facing):** NEVER signal building-to-be-acquired. The Mellanox/Run:ai
"NVIDIA buys the leader" point stays INTERNAL only, do not put it in any investor
answer. Use the CUDA framing instead (NVIDIA builds the platform, ecosystem builds the
layers). Interview: if pushed "NVIDIA contributes to PyTorch," answer "they optimize
silicon for it but didn't originate it, Meta did; NVIDIA builds the platform, the
ecosystem builds the layers, this is a layer."

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
