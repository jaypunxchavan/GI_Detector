# GI Detector

A sub-$100 electrochemical sensor pipeline for in vitro glycemic index estimation.

## What this is

This project replicates the Magaletta in vitro GI pipeline -- enzymatic starch digestion, glucose measurement, ML-based prediction -- using commodity hardware instead of HPLC. The goal is to quantify how much accuracy is lost when you replace a $10,000 lab instrument with an $80 Arduino-based sensor.

## Methodology

**Phase 1: Hardware calibration (in progress)**
Arduino Uno + INA219 current sensor reads glucose oxidase reaction current from Contour Next glucometer strips. Calibrated against dextrose standards at 50–400 mg/dL.

**Phase 2: Digestion protocol**
Food sample dissolved in acetate buffer (pH 6.0), incubated with alpha-amylase at 37°C. Glucose readings at 0, 15, 30, 45, 60 min. Area under the glucose release curve (AUC) computed per food.

**Phase 3: Full data collection**
40 foods across low/medium/high GI categories. Two trials per food minimum.

**Phase 4: ML comparison**
Three models compared: macronutrients only (baseline), AUC only, macronutrients + AUC combined. The delta between baseline and combined quantifies the sensor's contribution.
