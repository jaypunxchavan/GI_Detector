import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date


BASE_DIR   = Path(__file__).resolve().parent.parent.parent
CAL_PATH   = BASE_DIR / "results" / "validation" / "calibration_curve.json"
DATA_PATH  = BASE_DIR / "data" / "experimental" / "digestion_results.csv"
FIG_DIR    = BASE_DIR / "results" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)


print("Loading calibration coefficients...")
with open(CAL_PATH) as f:
    cal = json.load(f)

slope     = cal["slope_mA_per_mgdL"]
intercept = cal["intercept_mA"]

print(f"  Slope: {slope}  Intercept: {intercept}  R²: {cal['r_squared']}")

if not cal["r_squared_passes"]:
    print("⚠  WARNING: Calibration R² is below 0.95. Results may be unreliable.")
    print("   Re-run calibration.py before collecting data.")

def to_mgdL(reading_mA):
    """Convert raw INA219 mA reading to glucose concentration in mg/dL."""
    return (reading_mA - intercept) / slope


TIME_POINTS = [0, 15, 30, 45, 60]  # minutes


# TRIAL BLOCK — edit this section before each run
# Enter the raw INA219 current readings (in mA) from the serial monitor
# for each time point, for both the test food and white bread reference.


TRIAL = {
    # metadata, edit as needed
    "food_name":          "white bread",          # e.g. "lentils", "banana"
    "brand":              "Pepperidge Farm",       # exact product name
    "trial_number":       1,                       # 1, 2, or 3
    "date":               str(date.today()),       # auto-filled
    "buffer_pH_reading":  "6.0",                   # from pH strip
    "vinegar_volume_mL":  "~12",                   # approx mL added
    "enzyme_product":     "TAPCRAFT Alpha Amylase",
    "enzyme_lot":         "unknown",               # from packaging
    "enzyme_quantity_g":  0.3,
    "water_bath_temp_start_C": 37.0,
    "water_bath_temp_end_C":   37.0,
    "notes":              "",                      # any deviations

    # raw readings in mA. Replace these with your actual serial monitor output.
    # Order: [T=0, T=15, T=30, T=45, T=60]

    "food_readings_mA":   [0.00, 0.00, 0.00, 0.00, 0.00],   # REPLACE
    "bread_readings_mA":  [0.00, 0.00, 0.00, 0.00, 0.00],   # REPLACE
}


food_mgdL  = [to_mgdL(r) for r in TRIAL["food_readings_mA"]]
bread_mgdL = [to_mgdL(r) for r in TRIAL["bread_readings_mA"]]

print(f"\n── Glucose concentrations (mg/dL) ───────────────────")
print(f"  Time (min):  {TIME_POINTS}")
print(f"  {TRIAL['food_name']:20s}: {[round(v,1) for v in food_mgdL]}")
print(f"  White bread:          {[round(v,1) for v in bread_mgdL]}")


def trapezoidal_auc(concentrations, times):
    """
    Compute area under curve using trapezoidal rule.
    AUC = sum of ((g[i] + g[i+1]) / 2) * (t[i+1] - t[i])
    Units: mg/dL * minutes
    """
    auc = 0.0
    for i in range(len(times) - 1):
        auc += ((concentrations[i] + concentrations[i+1]) / 2) * (times[i+1] - times[i])
    return auc

auc_food  = trapezoidal_auc(food_mgdL, TIME_POINTS)
auc_bread = trapezoidal_auc(bread_mgdL, TIME_POINTS)

print(f"\n── AUC Results ──────────────────────────────────────")
print(f"  {TRIAL['food_name']:20s} raw AUC: {auc_food:.2f} mg/dL·min")
print(f"  White bread           raw AUC: {auc_bread:.2f} mg/dL·min")

#normalization
if auc_bread <= 0:
    print("\n ERROR: White bread AUC is zero or negative.")
    print("   Check that bread_readings_mA contains real values.")
    auc_normalized = None
else:
    auc_normalized = (auc_food / auc_bread) * 100
    print(f"\n  Normalized AUC: {auc_normalized:.1f}")
   

    # Rough GI category prediction from normalized AUC
    # These thresholds are calibrated against validation foods
    # and will be refined after Phase 3 data collection
    if auc_normalized <= 60:
        predicted_category = "low GI (≤55)"
    elif auc_normalized <= 85:
        predicted_category = "medium GI (56–69)"
    else:
        predicted_category = "high GI (≥70)"

    print(f"  Predicted category: {predicted_category}")
    print(f"  Note: category thresholds are preliminary until ML model is trained on full dataset.")

# plotting the digestion curves
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(TIME_POINTS, food_mgdL, "o-", color="#1A5276", linewidth=2,
        markersize=7, label=f"{TRIAL['food_name']} (AUC={auc_food:.1f})")
ax.plot(TIME_POINTS, bread_mgdL, "s--", color="#E74C3C", linewidth=2,
        markersize=7, label=f"White bread reference (AUC={auc_bread:.1f})")

ax.fill_between(TIME_POINTS, food_mgdL, alpha=0.1, color="#1A5276")
ax.fill_between(TIME_POINTS, bread_mgdL, alpha=0.1, color="#E74C3C")

ax.set_xlabel("Time (minutes)", fontsize=11)
ax.set_ylabel("Glucose Concentration (mg/dL)", fontsize=11)
ax.set_title(
    f"Digestion Curve — {TRIAL['food_name']}  |  "
    f"Normalized AUC = {auc_normalized:.1f}" if auc_normalized else
    f"Digestion Curve — {TRIAL['food_name']}",
    fontsize=12
)
ax.set_xticks(TIME_POINTS)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
food_slug = TRIAL["food_name"].replace(" ", "_").replace(",", "")
fig_path  = FIG_DIR / f"digestion_curve_{food_slug}_trial{TRIAL['trial_number']}.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved: {fig_path}")

# append to CSV
row = {
    "food_name":              TRIAL["food_name"],
    "brand":                  TRIAL["brand"],
    "trial_number":           TRIAL["trial_number"],
    "date":                   TRIAL["date"],
    "buffer_pH":              TRIAL["buffer_pH_reading"],
    "enzyme_product":         TRIAL["enzyme_product"],
    "enzyme_lot":             TRIAL["enzyme_lot"],
    "enzyme_quantity_g":      TRIAL["enzyme_quantity_g"],
    "water_bath_start_C":     TRIAL["water_bath_temp_start_C"],
    "water_bath_end_C":       TRIAL["water_bath_temp_end_C"],
    "T0_food_mA":             TRIAL["food_readings_mA"][0],
    "T15_food_mA":            TRIAL["food_readings_mA"][1],
    "T30_food_mA":            TRIAL["food_readings_mA"][2],
    "T45_food_mA":            TRIAL["food_readings_mA"][3],
    "T60_food_mA":            TRIAL["food_readings_mA"][4],
    "T0_bread_mA":            TRIAL["bread_readings_mA"][0],
    "T15_bread_mA":           TRIAL["bread_readings_mA"][1],
    "T30_bread_mA":           TRIAL["bread_readings_mA"][2],
    "T45_bread_mA":           TRIAL["bread_readings_mA"][3],
    "T60_bread_mA":           TRIAL["bread_readings_mA"][4],
    "auc_food_raw":           round(auc_food, 4),
    "auc_bread_raw":          round(auc_bread, 4),
    "auc_normalized":         round(auc_normalized, 2) if auc_normalized else "ERROR",
    "notes":                  TRIAL["notes"],
}

file_exists = DATA_PATH.exists()
with open(DATA_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)

print(f"Appended to: {DATA_PATH}")
print(f"\nTrial complete")
print(f"  Food: {TRIAL['food_name']} (Trial {TRIAL['trial_number']})")
print(f"  Normalized AUC: {auc_normalized:.1f}" if auc_normalized else "  Normalized AUC: ERROR")
print(f"  Date: {TRIAL['date']}")