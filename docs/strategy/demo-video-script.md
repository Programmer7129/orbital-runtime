# Steadstar — demo-only video script (~50s)

Pure MVP demo. No problem intro, no source recommendations. Record on the mission-control
dashboard (local file, or steadstar.vercel.app/demo once Vercel is connected).
Hit RESTART before recording. Speed 2× or 4×. Full-screen the browser.

---

"This is Steadstar running a real mission I trained on a rented NVIDIA L4. Three identical
GPT models, same seed, same data. The only difference is radiation and protection.

[point to the orbit panel]
Up here is the satellite's orbit. When it crosses the South Atlantic Anomaly, exposure
spikes and bit flips start landing. You can watch them count up here, and the number in
parentheses is how many hit inside the anomaly.

[point to the loss chart]
These are the training loss curves. Watch the unprotected run. It trains normally, and
then partway through, at step 179, a single bit flip corrupts it and the loss goes to NaN.
The run just dies. Its status flips to DIED right here.

[point to the protected run + tiles + log]
The protected run takes the same bombardment, 326 bit flips. Most are harmless. But when a
dangerous one lands, Steadstar catches it, rolls back to a verified checkpoint, and
replays. You can see it in these counters, 10 detected and rolled back, 141 steps replayed,
and in the recovery log scrolling here. It finishes all 300 steps, and its final loss
basically matches the clean, radiation-free baseline down here.

[close, full dashboard]
Same radiation. One run dies, one survives, for about 1.6% overhead. That's Steadstar."

---

## On-screen cues to hit (match narration to these)
- SAA entry: EXPOSURE flag elevates, "Upsets injected" tile starts climbing.
- Unprotected death: its loss curve spikes to NaN, status strip flips to DIED (~step 179).
- Protected recovery: "Detected → rolled back" ticks to 10, "Steps replayed" to 141,
  recovery log scrolls, protected run stays alive → completes 300/300.
- Footer: Protected val 2.6275 vs Clean val 2.5160 (basically matched).

## Numbers (match the committed dashboard)
- seed 3, 300 steps, 4 orbits, NVIDIA L4, calibrated 1e-7 upsets/bit-day.
- Unprotected: 128 upsets (114 in SAA) → NaN at step 179.
- Protected: 326 upsets (302 in SAA), 10 detected → 10 rolled back, 141 replayed, 300/300.
- Overhead: +1.6%.
- CAVEAT: 10-rolled-back / 141-replayed are the frozen M4b counts — correct for what's ON
  SCREEN. A fresh GPU rerun shifts those two slightly; the story doesn't change.
