# LANL landscape: current state of the field's founding institution (researched 2026-07-24)

## What they're doing NOW
- **Soft-error/SDC software research has gone quiet:** USRC publications stale since 2023;
  FTXS workshop (the field's home venue, LANL-run) CANCELLED for 2025; reliability lead
  Nathan DeBardeleben pivoted to AI-for-operations (queue prediction, repair-time
  trade-offs, EMC3 consortium). LANL vacated the software-FT territory exactly as
  orbital demand exploded.
- **But the hardware-testing infrastructure is being renewed for decades:** LAMP —
  $436M-$1.04B modernization of the LANSCE accelerator (CD-0 Nov 2024), operation
  through 2050. ICE House neutron beamline remains the industry-standard test site.
  Watch: construction outages late 2020s push testers to TRIUMF/ChipIr.
- **All-in on NVIDIA AI machines:** Venado (2,560 GH200s, 2024; now classified, runs
  OpenAI o-series for NNSA) → Mission + Vision on Vera Rubin, operational 2027. They
  will re-encounter GPU reliability at scale; historically they beam-test silicon
  pre-install.
- **Space side is ACTIVE but sensing-focused:** ESRA CubeSats (GTO radiation-belt
  mapping, NanoAvionics-built, 2024-26); GPS nuclear-detonation payloads (~2027).
  ZERO public engagement with orbital data centers.

## Key people
- **Heather Quinn (LANL ISR)** — the single closest person on Earth to our problem:
  space radiation effects on electronics, NSREC leadership, Xilinx Versal beam tests,
  neuromorphic-for-space (HPEC 2025), and heritage from the **Cibola Flight
  Experiment** — an SEU-recovering reconfigurable FPGA payload that FLEW: the direct
  intellectual ancestor of "software recovers from radiation on COTS hardware in
  orbit." TOP advisor/CSO-adjacent candidate; must cite Cibola in the paper/brief.
- Nathan DeBardeleben — fleet failure data + fault injection legacy (P-FSEFI);
  pivoted to AI-for-ops; plausible advisor for fleet-telemetry methodology.

## Free assets to exploit
- **P-FSEFI** — LANL's open-source parallel fine-grained soft-error fault injector
  (compare/cite vs our injector).
- **USRC public failure datasets** from production supercomputers (fleet-reliability
  methodology reference).
- **LANSCE Radiation Effects Testing handbook** (LAUR-19-30813, Wender) — canonical
  how-to for beam campaigns; read before writing the Crocker experiment plan.
- **NCSA "Story of Two GPUs" (arXiv:2503.11901, SC25)** — H100 memory MTBE is 3.2x
  WORSE than A100; ~5% overprovisioning needed for GPU failures. QUANTITATIVE WEDGE:
  each GPU generation gets less reliable while nobody ships the software layer.

## Strategic implications
1. No active program competes with us; the founding institution left the field as the
   market arrived. "LANL proved software resilience against radiation 20 years ago
   (ASC Q), then moved on; we're productizing their conclusion for orbit."
2. LANSCE = qualification partner #2 (neutrons/atmospheric; Crocker protons = LEO).
3. Recruiting: Quinn (space rad-effects) + LANL Ultrascale alumni pool.
4. Cite: ASC Q cosmic-ray study (Michalak), Cibola Flight Experiment, P-FSEFI,
   NCSA H100 study. Full links in agent transcript; headline links inline.
