# GI Detector
A sub-$100 electrochemical sensor pipeline for in vitro glycemic index estimation.

## What this is
This project replicates the Magaletta in vitro GI pipeline -- enzymatic starch digestion, glucose measurement, ML-based prediction -- using commodity hardware instead of HPLC. The goal is to quantify how much accuracy is lost when you replace a $10,000 lab instrument with an ~$200 Arduino-based sensor.

## Hardware architecture
Glucometer strip electrodes connect to a transimpedance amplifier (MCP6002 op-amp, channel A) that converts the microamp-level glucose oxidase reaction current into a 0–5V signal read by the Arduino's built-in ADC (analogRead, no external sensor IC required for the measurement itself). A 1MΩ feedback resistor sets the default gain, with 10MΩ, 200kΩ, and 100kΩ as troubleshooting fallbacks. A trim potentiometer provides an adjustable 0–5V bias on the second electrode, since the strip's electrochemistry needs a bias point away from 0V to produce a glucose-dependent signal. Full circuit wiring, pinout, and a troubleshooting tree (gain, bias, contact, and polarity issues) are documented in `src/hardware/`.

An earlier version of this project used an INA219 current sensor wired directly across the strip electrodes. That architecture doesn't work: the INA219 is a bus-current monitor intended for amp-to-milliamp-scale measurements, not the 1–5 microamp signal this strip produces, and it sits below the sensor's resolution at any usable gain setting. The INA219 module is no longer part of the measurement circuit.

## Methodology

**Phase 1: Hardware calibration (in progress)**
Arduino Uno + MCP6002 transimpedance amplifier reads the glucose oxidase reaction current from Contour Next glucometer strips as a 0–5V signal (raw ADC counts, 0–1023). Calibrated against dextrose standards at 50–400 mg/dL, prepared in the same 0.15M sodium acetate buffer used for digestion, so the calibration matrix matches the measurement matrix.

**Phase 2: Digestion protocol**
Food sample dissolved in 0.15M sodium acetate buffer (pH 5.8–6.2), incubated with alpha-amylase at 37°C in a sous vide water bath. Glucose readings at 0, 15, 30, 45, 60 min. Area under the glucose release curve (AUC) computed per food, normalized against a simultaneous white bread reference (white bread = 100).

**Phase 3: Full data collection**
20 foods across low/medium/high GI categories, plus 3 validation foods (white bread, rolled oats, lentils) used to confirm the pipeline before scaling up. Two trials per food minimum.

**Phase 4: ML comparison**
Three models compared: macronutrients only (baseline, established at 60.8% LOO accuracy on a 79-food computational dataset), AUC only (sensor signal alone), macronutrients + AUC combined. The delta between baseline and combined quantifies the sensor's contribution.
