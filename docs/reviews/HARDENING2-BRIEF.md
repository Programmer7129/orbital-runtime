You are the hardening builder, session 2. Read docs/reviews/FIX-LIST.md (your scope),
PLAN.md, STATUS.md. Items 6, 7, 8, 13 are DONE (commits 178b4d2, 1d373a2, 7a25cb7).
Commit 6c2168d is WIP on item 5 (wall-field stripping) — audit and finish it.

VERIFIED FINDING from the orchestrator: with the WIP stripping in place, two
NO_OPEN=1 demo/run_demo.sh runs on MPS still produce DIFFERENT telemetry_data.js
sha256 (0da0a57f... vs dbf98d72...) — MPS float nondeterminism drifts the loss values
themselves, so byte-exactness is impossible on MPS. Therefore: verify whether
byte-exactness holds on CPU (--device cpu, wall fields stripped); reword
README/run_demo.sh to semantic determinism on MPS (same death step, upsets,
rollbacks, event structure) + byte-exactness only where verified.

Then execute the REMAINING items in priority order:
- 1: detect_eval rescoring (TP requires first detection >= corruption_step) +
  clean-run FP rate as precision proxy + Clopper-Pearson 95% CIs on every ratio +
  regenerate detect-eval.json from current code
- 2: val-loss column in wall-clock table + band-top degradation disclosure
- 3: rewrite protect_overhead_calibrated.py with >=5 interleaved repeats + A/A
  control (ready for next GPU session); relabel committed L4 numbers as
  "single-seed, 2-repeat indicative"
- 4: ECC threat-model-inversion paragraph in README limitations
- then 9-12, 14-20 per FIX-LIST.md

Rules unchanged: regenerate all changed metrics from actual CPU/MPS reruns, never
hand-edit; no CUDA attempts (GPU is gone); preserve the honesty culture — never
soften a disclosed weakness; note each fix as "found by hostile review, fixed" in
STATUS.md; keep the suite green with regression tests for every code fix; commit in
logical chunks ("hardening: ..." prefixes); final STATUS.md entry summarizing what
changed and what remains disclosed-not-fixed; then stop.
