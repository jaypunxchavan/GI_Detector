"""
Usage:
    python auc_calc.py trial_data/lentils.json
    python auc_calc.py trial_data/white_bread.json

Reads a food JSON file (all 3 trials) and experiment_constants.json,
computes normalized AUC for each trial, plots digestion curves,
and appends results to the master CSV.

The primary result of this project is Spearman correlation between
mean normalized AUC (across 3 trials) and published GI value.
All five food files must be processed before running the final analysis.
"""

import json
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR       = Path(__file__).resolve().parent
CONSTANTS_PATH = BASE_DIR / "trial_data" / "experiment_constants.json"
CAL_PATH       = BASE_DIR / "results" / "validation" / "calibration_curve.json"
DATA_PATH      = BASE_DIR / "results" / "digestion_results.csv"
FIG_DIR        = BASE_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

if len(sys.argv) < 2:
    print("Usage: python auc_calc.py trial_data/<food>.json")
    sys.exit(1)

food_path = Path(sys.argv[1])
if not food_path.exists():
    print(f"ERROR: File not found: {food_path}")
    sys.exit(1)

with open(food_path) as f:
    food = json.load(f)

with open(CONSTANTS_PATH) as f:
    constants = json.load(f)

with open(CAL_PATH) as f:
    cal = json.load(f)

slope     = cal["slope_adc_per_mgdL"]
intercept = cal["intercept_adc"]
print(f"Calibration: slope={slope}, intercept={intercept}, R²={cal['r_squared']}")
if not cal["r_squared_passes"]:
    print("WARNING: Calibration R² below 0.95. Re-run calibration.py before proceeding.")

TIME_POINTS  = constants["digestion_time_points_min"]       # [0, 15, 30, 60, 120]
STRIP_MIN    = constants["strip_detection_range_mgdL"][0]   # 40
STRIP_MAX    = constants["strip_detection_range_mgdL"][1]   # 600
FOOD_NAME    = food["food_name"]
PUBLISHED_GI = food["published_gi_value"]


def to_mgdL(reading_adc):
    """Convert raw sensor current reading (adc) to glucose concentration (mg/dL).
    Reliable range: 40-600 mg/dL per strip specification.
    """
    return (reading_adc - intercept) / slope


def trapezoidal_auc(concentrations, times):
    """Area under glucose-release curve via trapezoidal rule.
    Units: mg/dL * minutes.
    """
    auc = 0.0
    for i in range(len(times) - 1):
        auc += ((concentrations[i] + concentrations[i+1]) / 2) * (times[i+1] - times[i])
    return auc


def check_range(values, trial_num, label):
    for t, v in zip(TIME_POINTS, values):
        if v < STRIP_MIN or v > STRIP_MAX:
            print(f"  WARNING Trial {trial_num} {label}: T={t}min reading {v:.1f} mg/dL "
                  f"outside reliable range ({STRIP_MIN}-{STRIP_MAX} mg/dL)")



print(f"Food: {FOOD_NAME}| Published GI: {PUBLISHED_GI}")

trial_results = []
fig, ax = plt.subplots(figsize=(8, 5))
colors_food  = ["#1A5276", "#2E86C1", "#85C1E9"]
colors_bread = ["#C0392B", "#E74C3C", "#F1948A"]

for i, trial_key in enumerate(["trial_1", "trial_2", "trial_3"], start=1):
    trial = food[trial_key]

    if None in trial["food_readings_adc"] or None in trial["bread_readings_adc"]:
        print(f"\nTrial {i}: SKIPPED — readings not yet filled in.")
        continue

    food_mgdL  = [to_mgdL(r) for r in trial["food_readings_adc"]]
    bread_mgdL = [to_mgdL(r) for r in trial["bread_readings_adc"]]

    check_range(food_mgdL,  i, "food")
    check_range(bread_mgdL, i, "bread")

    auc_food  = trapezoidal_auc(food_mgdL,  TIME_POINTS)
    auc_bread = trapezoidal_auc(bread_mgdL, TIME_POINTS)

    if auc_bread <= 0:
        print(f"\nTrial {i}: ERROR — white bread AUC is zero or negative. Check readings.")
        continue

    auc_normalized = (auc_food / auc_bread) * 100

    print(f"Trial {i}  ({trial['date']})")
    print(f"  pH: {trial['buffer_pH']}  |  "
          f"Temp: {trial['water_bath_temp_start_C']} -> {trial['water_bath_temp_end_C']} °C")
    print(f"  Time points (min):   {TIME_POINTS}")
    print(f"  {FOOD_NAME:20s}: {[round(v,1) for v in food_mgdL]} mg/dL")
    print(f"  White bread: {[round(v,1) for v in bread_mgdL]} mg/dL")
    print(f"  Food AUC: {auc_food:.2f} mg/dL·min")
    print(f"  Bread AUC: {auc_bread:.2f} mg/dL·min")
    print(f"  Normalized AUC: {auc_normalized:.1f}")
    if trial["notes"]:
        print(f"  Notes: {trial['notes']}")

    trial_results.append({
        "trial":          i,
        "date":           trial["date"],
        "buffer_pH":      trial["buffer_pH"],
        "temp_start_C":   trial["water_bath_temp_start_C"],
        "temp_end_C":     trial["water_bath_temp_end_C"],
        "food_mgdL":      food_mgdL,
        "bread_mgdL":     bread_mgdL,
        "auc_food":       auc_food,
        "auc_bread":      auc_bread,
        "auc_normalized": auc_normalized,
        "notes":          trial["notes"],
        "food_raw_adc":    trial["food_readings_adc"],
        "bread_raw_adc":   trial["bread_readings_adc"],
    })

    ax.plot(TIME_POINTS, food_mgdL, "o-", color=colors_food[i-1], linewidth=2,
            markersize=6, label=f"{FOOD_NAME} trial {i} (norm. AUC={auc_normalized:.1f})")
    ax.plot(TIME_POINTS, bread_mgdL, "s--", color=colors_bread[i-1], linewidth=1.5,
            markersize=5, alpha=0.6, label=f"White bread ref trial {i}")

if not trial_results:
    print("No completed trials to summarize.")
    sys.exit(0)

normalized_aucs = [r["auc_normalized"] for r in trial_results]
mean_auc = np.mean(normalized_aucs)
std_auc  = np.std(normalized_aucs)
cv_pct   = (std_auc / mean_auc) * 100 if mean_auc > 0 else float("nan")

print(f"SUMMARY: {FOOD_NAME}")
print(f"  Trials completed:    {len(trial_results)} / 3")
print(f"  Normalized AUCs:     {[round(x,1) for x in normalized_aucs]}")
print(f"  Mean normalized AUC: {mean_auc:.1f}")
print(f"  Std dev:             {std_auc:.1f}")
print(f"  CV%:                 {cv_pct:.1f}%  (trial-to-trial variability)")
print(f"  Published GI:        {PUBLISHED_GI}")
print(f"  Note: primary result is Spearman rho across all 5 foods,")
print(f"        computed in final_analysis.py after all foods are done.")

ax.set_xlabel("Time (minutes)", fontsize=11)
ax.set_ylabel("Glucose Concentration (mg/dL)", fontsize=11)
ax.set_title(
    f"Digestion Curves — {FOOD_NAME}\n"
    f"Mean Normalized AUC = {mean_auc:.1f} ± {std_auc:.1f}  |  Published GI = {PUBLISHED_GI}",
    fontsize=11
)
ax.set_xticks(TIME_POINTS)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()

food_slug = FOOD_NAME.replace(" ", "_")
fig_path  = FIG_DIR / f"digestion_curve_{food_slug}.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved figure: {fig_path}")

# --- Append to master CSV ---
# One row per trial. final_analysis.py reads this CSV to compute
# mean normalized AUC per food and run the Spearman correlation.
file_exists = DATA_PATH.exists()
with open(DATA_PATH, "a", newline="") as f:
    fieldnames = [
        "food_name", "published_gi_value", "trial", "date",
        "buffer_pH", "temp_start_C", "temp_end_C",
        "T0_food_adc",  "T15_food_adc",  "T30_food_adc",  "T60_food_adc",  "T120_food_adc",
        "T0_bread_adc", "T15_bread_adc", "T30_bread_adc", "T60_bread_adc", "T120_bread_adc",
        "auc_food_raw", "auc_bread_raw", "auc_normalized", "notes"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    for r in trial_results:
        writer.writerow({
            "food_name":          FOOD_NAME,
            "published_gi_value": PUBLISHED_GI,
            "trial":              r["trial"],
            "date":               r["date"],
            "buffer_pH":          r["buffer_pH"],
            "temp_start_C":       r["temp_start_C"],
            "temp_end_C":         r["temp_end_C"],
            "T0_food_adc":         r["food_raw_adc"][0],
            "T15_food_adc":        r["food_raw_adc"][1],
            "T30_food_adc":        r["food_raw_adc"][2],
            "T60_food_adc":        r["food_raw_adc"][3],
            "T120_food_adc":       r["food_raw_adc"][4],
            "T0_bread_adc":        r["bread_raw_adc"][0],
            "T15_bread_adc":       r["bread_raw_adc"][1],
            "T30_bread_adc":       r["bread_raw_adc"][2],
            "T60_bread_adc":       r["bread_raw_adc"][3],
            "T120_bread_adc":      r["bread_raw_adc"][4],
            "auc_food_raw":       round(r["auc_food"], 4),
            "auc_bread_raw":      round(r["auc_bread"], 4),
            "auc_normalized":     round(r["auc_normalized"], 2),
            "notes":              r["notes"],
        })

print(f"Appended {len(trial_results)} row(s) to: {DATA_PATH}")
