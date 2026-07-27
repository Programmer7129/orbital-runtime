# Steadstar — demo video script (~1:45)

Story arc: why-now → the gap → the product as the answer → see it live.
Record on the mission-control dashboard (local file, or steadstar.vercel.app/demo).
Hit RESTART before recording. Speed 2× or 4×. Full-screen the browser.
Plain and measured throughout. The facts carry the tension; no dramatic lines.

---

## 1. WHY WE'RE HERE — set the stage (~45s)

"AI is running out of power on Earth. The models keep getting bigger, and the datacenters
training them are already pushing against the limits of the power grid. So the industry is
looking to orbit, where solar power is abundant and runs around the clock. That's why
Google, SpaceX, and others are targeting real data centers in orbit as early as 2027.

There's one thing in the way. The chips that make this worth doing, commercial GPUs like
NVIDIA's, were built for climate-controlled rooms on the ground, not for space. Up there,
radiation constantly flips bits inside the silicon, and a single flip can silently corrupt
a result or kill a training run worth tens of millions of dollars, on a GPU nobody can
physically reach. The old fix, radiation-hardened chips, runs about a million times
slower, which defeats the point of flying a GPU at all.

So the whole industry is walking into one question: how do you let a normal GPU survive
orbit? That's what Steadstar does. Let me show you."

## 2. WHAT YOU'RE LOOKING AT — orient the viewer (~30s)

"This is a real run on a rented NVIDIA L4 GPU. The orbit around Earth is simulated, but
everything happening to the chip is real. The red stretch over one part of the orbit is the
South Atlantic Anomaly, the region where Earth's inner radiation belt dips closest to the
surface, because our magnetic field is tilted and offset from the center of the planet. A
satellite passing through it takes a sharp spike in radiation, so that's where most bit
flips happen. Every flash on the screen is one real bit flip hitting the chip, at the
actual rate a GPU would see in that orbit.

I'm training three identical GPT models side by side, same seed, same data. The only thing
different between them is radiation and protection. On the right
you can watch their training loss, one protected, one unprotected, and a clean,
radiation-free baseline. And these tiles keep count as it happens: bit flips landing, how
many were caught and rolled back, and steps replayed."

## 3. THE CRUX — one dies, one survives (~50s)

"Now watch what radiation does. The unprotected run trains normally, and then at step 179 a
single bit flip lands in a spot that matters, and its loss jumps to NaN. The run is dead,
and its status flips to DIED. In orbit, that's a training run worth millions, gone, on a
chip nobody can reach.

But not every hit is fatal, and this is the key to how Steadstar works. The model is
millions of floating-point numbers, and in each one, most of the bits only fine-tune the
value. When radiation flips one of those, the number barely changes, and training absorbs
it like a bit of noise. That's the large majority of these 326 hits, and there's no reason
to spend anything correcting them. The dangerous ones land on the few bits that control
the number's magnitude, its exponent. Flip one of those and a tiny weight can suddenly
become an enormous value, which either blows the run up into a NaN, like you just saw, or
silently corrupts the results without any warning.

So the protected run takes the same 326 hits, ignores the harmless ones, and the moment a
dangerous one lands, it catches it, rolls back to a verified checkpoint from just before,
and replays from there. You can see it happen: 10 caught and rolled back, 141 steps
replayed, each one logged here. The run survives all 300 steps and finishes at the same
loss as the clean baseline.

Same orbit, same radiation. The unprotected run dies, the protected one lives, and the
protection costs 1.6% in speed. That's Steadstar."

---

## On-screen cues (match narration to these)
- SAA entry: red arc on the orbit, EXPOSURE flag elevates, "Upsets injected" tile climbs.
- Unprotected death: loss curve spikes to NaN, status strip flips to DIED (~step 179).
- Protected recovery: "Detected → rolled back" → 10, "Steps replayed" → 141, log scrolls,
  protected run stays alive → completes 300/300.
- Footer: Protected val 2.6275 vs Clean val 2.5160 (matched).

## Numbers (match committed dashboard)
- seed 3, 300 steps, 4 orbits, NVIDIA L4, calibrated 1e-7 upsets/bit-day.
- Unprotected: 128 upsets (114 in SAA) → NaN at step 179.
- Protected: 326 upsets (302 in SAA), 10 detected → 10 rolled back, 141 replayed, 300/300.
- Overhead: +1.6%.
- CAVEAT: 10-rolled-back / 141-replayed are frozen M4b counts — correct for what's ON
  SCREEN; a fresh GPU rerun shifts those two slightly, story unchanged.
