import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = BASE_DIR / "data" / "experimental" / "calibration_readings.csv"
FIG_DIR = BASE_DIR / "results" / "figures"
VAL_DIR = BASE_DIR / "results" / "validation"

for d in [FIG_DIR, VAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
reading_cols = [c for c in df.columns if c.startswith("reading_")]
df["mean_reading_mA"] = df[reading_cols].mean(axis=1)
df["std_reading_mA"] = df[reading_cols].std(axis=1)

print(f" Concentrations: {df['concentration_mgdL'].tolist()} mg/dL")
print(f" Mean readings: {df['mean_reading_mA'].round(4).tolist()} mA")


X = df["concentration_mgdL"].values.reshape(-1, 1)
y = df["mean_reading_mA"].values

model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print("Calibration results:")
print(f" Slope:     {slope:.6f} mA per mg/dL")
print(f" Intercept: {intercept:.6f} mA")
print(f" R²:        {r2:.4f}")


if r2 >= 0.95:
    print(f" R² passes threshold (>=0.95). Proceed with data collection.")
else:
    print(" !! R² BELOW 0.95. Do not proceed with data collection.")
    print("Check strip contact consistency and wiring before retrying.")


fig, ax = plt.subplots(figsize=(7, 5))

ax.errorbar(
    df["concentration_mgdL"],
    df["mean_reading_mA"],
    yerr=df["std_reading_mA"],
    fmt="o", color="#1A5276", capsize=5, capthick=1.5,
    markersize=7, label="Standards (mean ± SD, n=3)"
)

x_line = np.linspace(0, 650, 200)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, color="#E74C3C", linewidth=1.5,
        label=f"Linear fit: y = {slope:.4f}x + {intercept:.4f}")

ax.set_xlabel("Glucose Concentration (mg/dL)", fontsize=11)
ax.set_ylabel("Sensor Current Reading (mA)", fontsize=11)
ax.set_title(f"Sensor Calibration Curve | R² = {r2:.4f}", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

color = "#1E8449" if r2 >= 0.95 else "#C0392B"
status = "PASS" if r2 >= 0.95 else "FAIL"
ax.text(0.97, 0.05, f"R² = {r2:.4f} [{status}]",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, color=color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color))

plt.tight_layout()
fig_path = FIG_DIR / "calibration_curve.png"
plt.savefig(fig_path, dpi=150)
plt.close()

# These are consumed by auc_calc.py at the start of every digestion run.
# Re-run this script any time the circuit is reassembled or strips change lot.
coefficients = {
    "slope_mA_per_mgdL": round(float(slope), 8),
    "intercept_mA": round(float(intercept), 8),
    "r_squared": round(float(r2), 6),
    "r_squared_passes": bool(r2 >= 0.95),
    "formula": "glucose_mgdL = (sensor_reading_mA - intercept) / slope",
    "concentration_range_mgdL": [40, 600],
    "n_standards": len(df),
    "note": (
        "Coefficients used by auc_calc.py to convert raw sensor readings "
        "to mg/dL. Re-run calibration if the circuit is reassembled or "
        "strips are from a new lot."
    )
}

json_path = VAL_DIR / "calibration_curve.json"
with open(json_path, "w") as f:
    json.dump(coefficients, f, indent=2)
print(f"Saved: {json_path}")

if not r2 >= 0.95:
    print(" Calibration failed. Do not proceed to food sample collection.")
else:
    print(" Calibration complete. Ready for food sample collection.")    
