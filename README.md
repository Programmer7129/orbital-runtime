# orbital-runtime (working name)

Software runtime that makes commercial GPUs survive radiation-induced faults in orbit:
fault injection calibrated to LEO radiation environments → detection → checkpointing →
invisible job recovery. "ECC memory for the orbital era, sold as software."

## Status

Pre-MVP. Planning happens in the lead Claude Code session; execution happens in a
builder teammate session working from `PLAN.md`.

## Layout

- `PLAN.md` — the implementation plan (source of truth for the builder session)
- `docs/research/` — synthesized research backing the plan (market, physics, tooling)

## Context

Target: YC application demo. Show a real PyTorch training run (nanoGPT-class) dying
from injected single-event upsets at published LEO rates, then surviving with our
runtime at low overhead, with a mission-control dashboard of upsets/detections/recoveries.
