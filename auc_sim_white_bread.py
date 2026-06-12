import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
CAL_PATH   = BASE_DIR / "results" / "validation" / "calibration_curve.json"
DATA_PATH  = BASE_DIR / "data" / "experimental" / "digestion_results_simulated.csv"
FIG_DIR    = BASE_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(CAL_PATH) as f:
    cal = json.load(f)
slope     = cal["slope_mA_per_mgdL"]
intercept = cal["intercept_mA"]
print(f"  Slope: {slope}  Intercept: {intercept}  R²: {cal['r_squared']}")

def to_mgdL(reading_mA):
    return (reading_mA - intercept) / slope

TIME_POINTS = [0, 15, 30, 45, 60]

TRIAL = {
    "food_name":               "white bread",
    "brand":                   "SIMULATED — Montebugnoli 2025 reference values",
    "trial_number":            1,
    "date":                    str(date.today()),
    "buffer_pH_reading":       "6.0 (simulated)",
    "vinegar_volume_mL":       "simulated",
    "enzyme_product":          "SIMULATED",
    "enzyme_lot":              "SIMULATED",
    "enzyme_quantity_g":       0.3,
    "water_bath_temp_start_C": 37.0,
    "water_bath_temp_end_C":   37.0,
    "notes":                   "SIMULATED DATA — pipeline validation only. Not real sensor readings.",
    "food_readings_mA":        [0.15, 0.52, 0.74, 0.85, 0.89],
    "bread_readings_mA":       [0.15, 0.52, 0.74, 0.85, 0.89],
}

food_mgdL  = [to_mgdL(r) for r in TRIAL["food_readings_mA"]]
bread_mgdL = [to_mgdL(r) for r in TRIAL["bread_readings_mA"]]

print(f"── Glucose concentrations (mg/dL))")
print(f"  Time (min):  {TIME_POINTS}")
print(f"  {TRIAL['food_name']:20s}: {[round(v,1) for v in food_mgdL]}")
print(f"  White bread:          {[round(v,1) for v in bread_mgdL]}")

def trapezoidal_auc(concentrations, times):
    auc = 0.0
    for i in range(len(times) - 1):
        auc += ((concentrations[i] + concentrations[i+1]) / 2) * (times[i+1] - times[i])
    return auc

auc_food  = trapezoidal_auc(food_mgdL, TIME_POINTS)
auc_bread = trapezoidal_auc(bread_mgdL, TIME_POINTS)
auc_normalized = (auc_food / auc_bread) * 100

print(f"AUC Results: ")
print(f"  White bread raw AUC:  {auc_food:.2f} mg/dL·min")
print(f"  Normalized AUC:       {auc_normalized:.1f}  (expected: 100.0)")

if abs(auc_normalized - 100.0) < 0.01:
    print(" Pipeline validation PASSED: white bread normalizes to 100.0 correctly.")
else:
    print(" Pipeline validation FAILED: check calibration coefficients.")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(TIME_POINTS, food_mgdL, "o-", color="#E74C3C", linewidth=2, markersize=7,
        label=f"White bread (AUC={auc_food:.1f}) — SIMULATED")
ax.fill_between(TIME_POINTS, food_mgdL, alpha=0.1, color="#E74C3C")
ax.set_xlabel("Time (minutes)", fontsize=11)
ax.set_ylabel("Glucose Concentration (mg/dL)", fontsize=11)
ax.set_title(f"Simulated Digestion Curve — White Bread  |  Normalized AUC = {auc_normalized:.1f}", fontsize=11)
ax.set_xticks(TIME_POINTS)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.97, 0.05, "SIMULATED DATA", transform=ax.transAxes, ha="right",
        va="bottom", fontsize=9, color="gray", style="italic")
plt.tight_layout()
fig_path = FIG_DIR / "digestion_curve_white_bread_sim_trial1.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved: {fig_path}")

row = {
    "food_name": TRIAL["food_name"], "brand": TRIAL["brand"],
    "trial_number": TRIAL["trial_number"], "date": TRIAL["date"],
    "buffer_pH": TRIAL["buffer_pH_reading"], "enzyme_product": TRIAL["enzyme_product"],
    "enzyme_lot": TRIAL["enzyme_lot"], "enzyme_quantity_g": TRIAL["enzyme_quantity_g"],
    "water_bath_start_C": TRIAL["water_bath_temp_start_C"],
    "water_bath_end_C": TRIAL["water_bath_temp_end_C"],
    "T0_food_mA": TRIAL["food_readings_mA"][0], "T15_food_mA": TRIAL["food_readings_mA"][1],
    "T30_food_mA": TRIAL["food_readings_mA"][2], "T45_food_mA": TRIAL["food_readings_mA"][3],
    "T60_food_mA": TRIAL["food_readings_mA"][4],
    "T0_bread_mA": TRIAL["bread_readings_mA"][0], "T15_bread_mA": TRIAL["bread_readings_mA"][1],
    "T30_bread_mA": TRIAL["bread_readings_mA"][2], "T45_bread_mA": TRIAL["bread_readings_mA"][3],
    "T60_bread_mA": TRIAL["bread_readings_mA"][4],
    "auc_food_raw": round(auc_food, 4), "auc_bread_raw": round(auc_bread, 4),
    "auc_normalized": round(auc_normalized, 2), "notes": TRIAL["notes"],
}
file_exists = DATA_PATH.exists()
with open(DATA_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)
