# GI Detector
A sub-$300 electrochemical sensor pipeline for in vitro glycemic index estimation.

## What this is
Tests whether a commodity glucometer strip, read by a custom Arduino circuit, can estimate glycemic index (GI) as accurately as lab-grade methods (HPLC, enzymatic assays) at a fraction of the cost. Food samples undergo simulated digestion, glucose release is tracked over time, and the resulting signal is correlated against each food's published GI value.

## Hardware
Glucometer strip electrodes connect to a transimpedance amplifier (MCP6002 op-amp) that converts the glucose oxidase reaction current into a 0–5V signal read by the Arduino's ADC. 1MΩ feedback resistor (10MΩ/200kΩ/100kΩ fallbacks), trim potentiometer for bias, ICL7660 negative-bias circuit included. Full wiring and troubleshooting docs in `src/hardware/`.

Uses TRUENESS glucometer strips (glucose oxidase, 40–600 mg/dL range) with a TRUENESS meter as ground truth during calibration.

## Methodology

**Phase 0: Pre-Procedure validation** — Circuit tested against known dextrose standards (50–400 mg/dL) in matched buffer.

**Phase 1: Digestion protocol** — Food standardized to ~0.5g available carbohydrate, ground and sieved, dissolved in ~20mL 0.15M sodium acetate buffer (pH 4.5–5.0). Alpha-amylase + glucoamylase added at T=0, incubated at 37°C. Glucose read at T=0/15/30/60 min. AUC normalized against a white bread reference (white bread = 100).

**Phase 2: Data collection** — 5 foods spanning the GI range (white bread 100, white rice ~73, oats ~50, lentils ~29, chickpeas ~28), 3 trials each.

**Phase 3: Analysis** — Spearman correlation between sensor AUC and published GI values, with confidence interval. Macronutrient-only baseline (60.8% LOO accuracy, 79-food dataset) retained as background context.

## Status
Nothing physically built or purchased yet. See `docs/` for full materials list and assembly protocol.
