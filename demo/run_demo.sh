#!/usr/bin/env bash
#
# run_demo.sh — the headline demo, end to end, on a laptop, in under 5 minutes.
#
#   1. train three IDENTICAL nanoGPT runs (seed 1337), differing only in
#      radiation + protection:
#        baseline     rate 0     protect off   -> the clean reference
#        unprotected  rate 3e-6  protect off   -> DIES (NaN) at step 141
#        protected    rate 3e-6  protect on    -> SURVIVES, matches baseline
#   2. bundle their telemetry into the dashboard's data file
#   3. open the split-screen mission-control dashboard
#
# Everything is seeded and deterministic: the same three curves, the same 49
# upsets, the same 7 rollbacks, every time. Nothing here rents a GPU, records
# video, or touches the calibrated constants -- that is M4b.
#
# Why the rate is elevated (3e-6, ~300x the calibrated 1e-9..1e-7 flight band):
# this demo model holds 7.8e7 resident bits against an H100's 6.4e11 -- four
# orders of magnitude fewer bits to hit, so the flight-band rate would deliver
# almost nothing in 200 steps. The band itself is asserted against real H100
# bit counts in tests/test_flux.py; M4b re-measures at real scale on a GPU.

set -euo pipefail

# --- locate the repo root (this script lives in demo/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# --- pick an interpreter: prefer the project venv, fall back to python3 ---
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
  echo "note: .venv not found, using '$PY' (run 'make venv install' for the pinned env)"
fi

SEED=1337
STEPS=200
ORBITS=2
RATE=3e-6
COMMON=(--workload nanogpt --orbits "$ORBITS" --seed "$SEED" --steps "$STEPS" --log-every 1)

# --- ensure the training corpus is present ---
CORPUS="demo/workloads/nanogpt/data/input.txt"
if [[ ! -f "$CORPUS" ]]; then
  echo ">> fetching Shakespeare corpus (one-time)…"
  make data
fi

bar() { printf '%s\n' "========================================================================"; }

bar
echo "ORBITAL RUNTIME — headline demo   (seed $SEED, $STEPS steps, $ORBITS orbits, rate $RATE)"
bar

echo
echo ">> [1/3] CLEAN BASELINE — no radiation"
"$PY" -m orbital_runtime.run "${COMMON[@]}" --rate 0 --protect off --tag baseline

echo
echo ">> [2/3] UNPROTECTED under radiation — expected to DIE"
# A dead run exits non-zero, and that is the whole point; don't let it abort us.
set +e
"$PY" -m orbital_runtime.run "${COMMON[@]}" --rate "$RATE" --protect off --tag unprotected
UNPROT_RC=$?
set -e
if [[ "$UNPROT_RC" -eq 0 ]]; then
  echo "!! WARNING: the unprotected run did NOT die (rc=0). The demo story depends"
  echo "!!          on it dying — check the seed/rate before showing this."
fi

echo
echo ">> [3/3] PROTECTED under identical radiation — expected to SURVIVE"
"$PY" -m orbital_runtime.run "${COMMON[@]}" --rate "$RATE" --protect on --tag protected

echo
echo ">> bundling telemetry into the dashboard…"
# No wall-clock timestamp here on purpose: with a fixed seed the whole pipeline
# is byte-for-byte reproducible, so re-running the demo produces an identical
# telemetry_data.js (no spurious git churn, nothing that could flake on stage).
"$PY" demo/dashboard/build.py --seed "$SEED"

DASH="$ROOT/demo/dashboard/index.html"
bar
echo "DONE. Open the split-screen dashboard:"
echo "  $DASH"
bar

# --- open the dashboard (best-effort; the path above always works by hand) ---
if [[ "${NO_OPEN:-}" != "1" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "$DASH"            # macOS
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$DASH"       # Linux
  fi
fi
