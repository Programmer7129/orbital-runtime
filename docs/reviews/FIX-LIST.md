# Consolidated fix list from two hostile reviews (2026-07-18)

Full reviews: methodology-review.md, artifact-review.md (same directory).
Priority order for the fix session. Items marked [DOC] are README/STATUS wording;
[CODE] are code changes; [BOTH] are both.

## FATAL — must fix before professor outreach

1. [CODE] detect_eval scoring bug: TP must require first detection >= corruption_step.
   Report clean-run FP rate as the headline precision proxy (irradiated-cohort
   "precision 1.00" is vacuous — tn=fp=0 by construction). Add Clopper-Pearson 95%
   CIs to every ratio ("6/6, 95% CI 0.54-1.0"). Regenerate detect-eval.json with
   current code (it's stale: seed 11 guard latency 92->77, seed 3 flips 98->99).
2. [BOTH] "Survived" hides degradation: protect_overhead_l4.json 1e-7 row has
   final_val 4.038 vs baseline ~2.52. Add val-loss column to the wall-clock README
   table; state band-top survivors are degraded (M3 sub-detection-floor mechanism).
3. [BOTH] protect_overhead_calibrated.py violates design rule 4: REPEATS=2, no A/A,
   no interleaving. Rewrite the script with >=5 interleaved repeats + A/A control
   (ready for next GPU session); relabel current README numbers as "single-seed,
   2-repeat indicative — proper controlled rerun pending".
4. [DOC] ECC threat-model inversion paragraph: headline is ecc_off single-bit DRAM
   flips; deployments fly ECC-on where residual channels (MBU leakage, SRAM/logic,
   SEFI) are exactly the uncited/off-by-default ones. Own it: "ecc_off is the
   calibratable proxy regime; the ECC-on channels are what beam validation
   calibrates." Add to README limitations.
5. [BOTH] Byte-identical claim is false on MPS: (a) strip wall-clock fields from
   the dashboard bundle in build.py; (b) add checkpoint_wall_s to
   telemetry.WALL_CLOCK_FIELDS (its absence breaks strip_wall determinism proof —
   verified 1 differing event); (c) reword README/run_demo.sh to semantic
   determinism (same death step/upsets/rollbacks), byte-exactness CPU-only if it
   actually holds there (verify).
6. [CODE] Rollback resume off-by-one: _finish_rollback returns ck.step but the
   checkpoint is post-step state -> step s update applied twice. Resume at
   ck.step + 1. Reconcile the "+112 replayed" vs "replayed 105" counters (delta =
   n_rollbacks). Also fix "(+1 replayed)" on dead unprotected runs (M1).

## HIGH — fix or explicitly disclose

7. [CODE] Trusted-snapshot ordering: detector.observe() must run BEFORE
   recovery.after_step (save) and abft.refresh_checksums(), so a detected-bad step
   is neither checkpointed as verified=True nor laundered into the trusted
   checksum. Post-rollback first-step blind window (env.advance fires before
   forward; reset() clears _trusted; "no radiation yet" fallback false there):
   fix if tractable, else disclose as honesty flag with the one-sampled-forward
   escape probability.
8. [CODE] LAG_UNLOCALISED=25 justified by false "covers measured worst case" —
   guards max latency is 92 (committed data). Raise to cover it or make it derive
   from measured max; fix comment. Disclose the 40-step guard warmup blindness
   after every rollback (Detector.reset()).
9. [DOC] Band-edge disclosure: unprotected death only demonstrated at 1e-7 (top of
   cited band; flight data for modern memory sits 1e-9..1e-8; NEPP figures are
   design criteria not observations). State: at 1e-8/1e-9 unprotected survives the
   tested mission length.
10. [DOC] Injection surface + timing: flips hit params+optimizer only, between
    optimizer.step() and next forward (max ABFT visibility); gradients/activations
    (headline)/mid-kernel never struck. State explicitly; frame as beam questions.
11. [DOC] Narrow novelty claim to "position-aware protection scheduling for
    general-purpose GPU training runtimes" (SAA-aware instrument safing is
    decades-old practice). Fix in README + docstrings.
12. [DOC] Oracle scope: requires bit-exact determinism (not enforced via
    torch.use_deterministic_algorithms — note), loss-visible divergence within
    120-step horizon only, no oracle post-first-rollback.
13. [CODE] Checkpoint completeness: save CUDA/MPS RNG state, not just CPU
    (silently breaks exact replay for dropout>0 workloads). Cheap fix.

## MEDIUM — polish

14. [DOC] fp32-only disclosure (bit-30 analysis, tolerances, all numbers; bf16/fp16
    differ materially — V-ABFT is the roadmap).
15. [DOC] NVIDIA 5x citation scoped: validated SASS-level models, not tensor-level
    parameter injection — soften blanket legitimation.
16. [DOC] SAA model idealization: identical fixed-phase transit every orbit is
    geographically impossible (real: precessing ground track, subset of orbits);
    track.py discloses but README should too, since it flatters adaptive vigilance.
17. [CODE] abft.py: remove dead _tolerance() duplicate.
18. [DOC] Same-section apples-to-oranges: label block_size=64 vs 256 configs in the
    L4 tables; keep device-scale flux (H100 bits) visually separate from demo-scale
    (8.19e9 bits) numbers.
19. [BOTH] Bench provenance: make bench must regenerate committed filenames with
    the exact flags; record model-size + config metadata inside the JSONs.
20. [DOC] Trivia: README 263-vs-262 test count; bench/results vs bench_out path
    story; never attach "HBM" to L4 (GDDR6) measurements; pin/record torch version
    with committed results.
