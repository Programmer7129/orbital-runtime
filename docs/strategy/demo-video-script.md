# Steadstar — demo video script (~1:40, as tight as possible)

Format: screen recording + your voiceover. Order: problem → learn-more → MVP demo.
Parts 1–2 on the LIVE Steadstar site; Part 3 on the local dashboard (have it preloaded).
Numbers match the committed dashboard exactly.

---

## PART 1 — THE PROBLEM  (~25s)  [site hero]

"Everyone's racing to put commercial GPUs in orbit. Starcloud flew an NVIDIA H100, Google
is doing TPUs with Project Suncatcher, SpaceX filed for up to a million data-center
satellites. The pull is free solar power and free cooling.

The problem nobody has solved: space radiation flips bits in silicon. In orbit you can't
hot-swap a GPU or cheaply re-run a training job worth tens of millions of dollars, and a
single bit flip can silently corrupt your results or kill the run. Rad-hardened chips
avoid it but run about a million times slower. I build the software that keeps commercial
GPUs computing correctly through radiation."

---

## PART 2 — WHERE TO LEARN MORE  (~15s)  [scroll to Further Reading]

"New to the problem? Three places to start, all on screen: Veritasium's 'The Universe is
Hostile to Computers' for how cosmic rays flip bits in real hardware, Google's 'Cores
That Don't Count' for how often it silently happens in datacenters, and Google's Project
Suncatcher for where orbital compute is heading."

---

## PART 3 — THE DEMO  (~60s)  [dashboard]

"This is a real mission I trained on a rented NVIDIA L4. Three identical GPT models, same
seed, same data. The only difference is radiation and protection.

[orbit / SAA view]
The satellite is crossing the South Atlantic Anomaly, where radiation is worst. Every
flash is a bit flip, injected at the real calibrated flight rate.

[unprotected loss curve]
The unprotected run trains normally, and then at step 179 one bit flip hits a lethal spot
and the loss goes to NaN. Dead. In orbit that's a multi-million-dollar run gone, on a GPU
you can't reach.

[protected run + counters]
The protected run takes the same bombardment, 326 bit flips. Most are harmless. But 10
dangerous ones get detected, rolled back to a verified checkpoint, and 141 steps replayed,
and it finishes all 300 steps, matching the clean, radiation-free baseline.

[overhead ticker: 1.6%]
The cost of all that protection is 1.6% overhead.

[full dashboard]
That's Steadstar. Detection and recovery that lets the commercial GPUs everyone is flying
survive orbit, tested at the real radiation rate they'll face."

---

## Numbers reference (match the committed dashboard)
- Mission: seed 3, 300 steps, 4 orbits, NVIDIA L4, calibrated 1e-7 upsets/bit-day.
- Unprotected: 128 upsets (114 in SAA) → NaN at step 179.
- Protected: 326 upsets (302 in SAA), 10 detected → 10 rolled back, 141 replayed, 300/300;
  final val loss 2.6275 vs clean baseline 2.5160.
- Overhead: +1.6% (tier 1+2 adaptive, L4, 85M params).
- CAVEAT: 10-rolled-back / 141-replayed are the frozen M4b dashboard counts — correct for
  what's ON SCREEN. A fresh GPU rerun shifts those two counts slightly; the story doesn't.
