# Beam-data calibration audit (2026-07-24)

Audit of ALL published radiation-beam measurements against this simulator's assumptions.
Question: can published data validate us, and is our novelty claim actually unclaimed?

## Headline findings

1. **Training-under-beam is VERIFIED UNCLAIMED.** Every workload-under-beam paper runs
   inference or kernels (Suncatcher transformer inference; Jetson NN inference; Rech-group
   YOLO/ResNet on GPUs). No published campaign irradiates a device during TRAINING
   (forward+backward+optimizer) with application-level outcome telemetry. Our proposed
   Crocker experiment's novelty holds as of this search.
2. **Base band 1e-9..1e-7 upsets/bit-day: order-of-magnitude consistent.** Terrestrial
   SRAM SER anchor (~2.4e-11 upsets/bit-day) makes our band 40-4000x terrestrial —
   plausible for LEO trapped protons behind shielding. No published source pins it
   directly (Suncatcher reports per-CHIP not per-bit cross-sections; NVIDIA papers
   redact absolute FIT as business-sensitive) — hence our beam campaign.
3. **SAA 75x multiplier: conservative.** Literature range for SAA trapped-proton
   enhancement is ~100-1000x; beam tests cannot validate this (it's an orbital flux
   parameter — AP8/AP9 models, not cyclotrons).
4. **CALIBRATION ERRORS FOUND (to fix, with citations now available):**
   - **SEFI channel off is contradicted by every source that measured it.**
     Suncatcher: SEFI sigma = 2e-11 cm²/chip (~1 per 5 krad). All three Jetson proton
     campaigns: reboot cross-section EXCEEDS application-error cross-section. NSREC'21
     GPU study: with ECC ON, functional-interrupt (DUE) rate exceeds silent-corruption
     rate by 2.2-2.7x. SEFI must default ON, calibrated to Suncatcher's number.
   - **Independent single-bit flips mismodel HBM.** MICRO'21 (V100 HBM2, ChipIR):
     31.5% of SEUs are multi-bit; ~75% of those byte-contiguous; broadest event hit
     5,359 memory entries. A per-bit-independent model undercounts correlated
     corruption and overstates ECC effectiveness.
   - **ECC should SHIFT upsets SDC→DUE, not just suppress:** ECC cuts SDC up to 21x
     but RAISES DUE up to 13.7x (NSREC'21). Model ECC as redistribution.
   - **No TID/aging term** (Suncatcher HBM irregularities at 2 krad cumulative;
     MICRO'21 displacement-damage weak cells) — disclosed gap, out of scope for a
     rate-only simulator, but must be stated.
5. **The two "citations owed" placeholders now have citable starting values:**
   - MBU fraction: MICRO'21's 31.5% multi-bit share (neutron, HBM2 — conditions
     caveat) replaces the invented DEFAULT_ECC_LEAK_FRACTION=0.02 as a cited anchor.
   - SEFI per-transit: derivable from Suncatcher 2e-11 cm²/chip + orbital flux model.

## Key extracted numbers

| Source | Quantity | Value |
|---|---|---|
| Suncatcher (arXiv 2511.19468; 67 MeV p+, Crocker) | SDC sigma | 6-9e-9 cm²/chip |
| Suncatcher | HBM UECC sigma | 3e-9 cm²/chip (203 events) |
| Suncatcher | SEFI sigma | 2e-11 cm²/chip (~1/5 krad) |
| Suncatcher | TID | no hard failure to 15 krad(Si); HBM irregularities from 2 krad |
| Suncatcher | dose-fluence conversion | 1 rad ≈ 7.9e6 p/cm² (67 MeV) |
| MICRO'21 HBM2 (doi 10.1145/3466752.3480111) | multi-bit SEU share | 31.5% (75% byte-contiguous) |
| NSREC'21 GPU DUE (arXiv 2108.00554) | ECC-ON DUE:SDC | 2.2-2.7x (DUE dominant) |
| NSREC'21 | ECC effect | SDC ÷21x, DUE ×up-to-13.7x |
| Jetson campaigns (REDW'21/'23, RADECS'24) | reboot vs app-error sigma | reboot dominant (exact values paywalled) |

## Consequences

- Brief/paper: add "validated against all published beam data at the rate level;
  outcome level is unmeasured — that is the experiment."
- M4c builder scope now includes: SEFI on by default (Suncatcher-calibrated), MBU
  clustering model (MICRO'21), ECC-as-redistribution, updated disclosures.
- Full source list and per-parameter verdict table in the audit agent transcript;
  citations inline above.
