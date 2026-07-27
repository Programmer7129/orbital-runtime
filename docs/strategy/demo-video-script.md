# Steadstar — demo video script (~75s)

Framing → screen orientation → the crux. Record on the mission-control dashboard
(local file, or steadstar.vercel.app/demo once Vercel is connected).
Hit RESTART before recording. Speed 2× or 4×. Full-screen the browser.

---

## 1. FRAMING — who it's for and why it's inevitable (~20s)

"Let me show you what Steadstar gives the companies racing to put data centers in space.
The hyperscalers, Google and SpaceX among them, are promising orbital compute as early as
2027, and the moment anyone gets GPUs running up there sustainably, they all hit the same
wall. Space radiation flips bits in silicon, and a single flip can silently corrupt or
kill a training run worth tens of millions of dollars, on a GPU you can't reach to reset
or swap out."

## 2. SCREEN ORIENTATION — what they're looking at (~25s)

"What you're looking at right now is a simulation of a real orbit around Earth. Notice the
red stretch over one part of the orbit. That's the South Atlantic Anomaly, where Earth's
inner radiation belt dips closest to the surface, because our magnetic field is tilted and
offset from the center of the planet. Any satellite passing through it takes a sharp spike
in radiation, and that's where most bit flips happen. Every flash you see is a real bit
flip landing on the chip, at the actual calibrated flight-band rate.

I'm running three identical GPT models here, same seed, same data. The only difference
between them is radiation and protection. On the right are their training loss curves,
protected, unprotected, and a clean baseline. And these tiles track what's happening live,
bit flips injected, how many were caught and rolled back, and steps replayed."

## 3. THE CRUX — one dies, one survives (~30s)

"Now here's the whole point. Watch the unprotected run. It trains normally, and then at
step 179 a single bit flip hits a lethal spot and its loss goes to NaN. It's dead, and its
status flips to DIED. In orbit, that's a multi-million-dollar run gone.

The protected run takes the exact same bombardment, 326 bit flips. Most are harmless, so
Steadstar leaves them alone. But the moment a dangerous one lands, it catches it, rolls
back to a verified checkpoint, and replays, 10 caught and rolled back, 141 steps replayed,
all of it in this recovery log. It finishes all 300 steps, and its final loss matches the
clean, radiation-free baseline.

Same radiation. One run dies, one survives, for 1.6% overhead. That's Steadstar."

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
