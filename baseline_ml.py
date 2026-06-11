import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
)
import joblib

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = "/Users/arnavchavan/Desktop/gi_sensor_project/data/processed/food_gi_dataset_v2.csv"
FIG_DIR    = BASE_DIR / "results" / "figures"
MODEL_DIR  = BASE_DIR / "results" / "models"
VAL_DIR    = BASE_DIR / "results" / "validation"

for d in [FIG_DIR, MODEL_DIR, VAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  {len(df)} foods loaded")
print(f"  GI distribution: {dict(df['gi_category'].value_counts())}")

FEATURES = ["carb_g", "fiber_g", "sugar_g", "fat_g", "protein_g"]
TARGET   = "gi_category"
CLASSES  = ["low", "medium", "high"]          # canonical order for plots

X = df[FEATURES].values
y = df[TARGET].values


models = {
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

loo = LeaveOneOut()



print("PHASE 0 BASELINE — Macronutrients Only (no sensor)")
print("=" * 55)

cv_results = {}
best_name, best_acc = None, 0.0

for name, model in models.items():
    acc_scores = cross_val_score(model, X, y, cv=loo, scoring="accuracy")
    f1_scores  = cross_val_score(model, X, y, cv=loo, scoring="f1_weighted")
    acc_mean   = acc_scores.mean()
    f1_mean    = f1_scores.mean()

    cv_results[name] = {"loo_accuracy": round(acc_mean, 4),
                        "loo_f1_weighted": round(f1_mean, 4)}

    print(f"\n{name}:")
    print(f"  LOO Accuracy:     {acc_mean:.3f} ± {acc_scores.std():.3f}")
    print(f"  LOO F1 (weighted): {f1_mean:.3f}")

    if acc_mean > best_acc:
        best_acc, best_name = acc_mean, name

print(f"\nBest model: {best_name} ({best_acc:.3f})")

best_model = models[best_name]
y_pred = cross_val_predict(best_model, X, y, cv=loo)

cm = confusion_matrix(y, y_pred, labels=CLASSES)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Baseline Confusion Matrix\n{best_name} — LOO CV (n={len(df)})",
             fontsize=11, pad=10)
plt.tight_layout()
cm_path = FIG_DIR / "baseline_confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\nSaved: {cm_path}")

#feature importance
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
feat_order  = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#4A90D9"] * len(FEATURES)
bars = ax.bar(
    [FEATURES[i] for i in feat_order],
    [importances[i] for i in feat_order],
    color=colors, edgecolor="white", linewidth=0.5
)
ax.set_title("Feature Importance — Baseline RF (macronutrients only)", fontsize=11)
ax.set_ylabel("Mean Decrease in Impurity")
ax.set_xlabel("Feature")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
for bar, i in zip(bars, feat_order):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{importances[i]:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
fi_path = FIG_DIR / "baseline_feature_importance.png"
plt.savefig(fi_path, dpi=150)
plt.close()
print(f"Saved: {fi_path}")


print("\n")
print("Feature Importance (Random Forest, fit on all data):")
for i in feat_order:
    bar = "█" * int(importances[i] * 40)
    print(f"  {FEATURES[i]:15s} {importances[i]:.3f}  {bar}")

#save the model
model_path = MODEL_DIR / "baseline_rf_model.joblib"
joblib.dump(rf, model_path)
print(f"\nSaved model: {model_path}")

#save it to JSON
results = {
    "phase": "Phase 0: Macronutrients Only Baseline",
    "n_foods": len(df),
    "class_distribution": {k: int(v) for k, v in df["gi_category"].value_counts().items()},
    "features_used": FEATURES,
    "cross_validation": "Leave-One-Out",
    "models": cv_results,
    "best_model": best_name,
    "best_loo_accuracy": round(best_acc, 4),
    "target_accuracy_with_sensor": 0.75,
    "note": (
        "This is the accuracy ceiling achievable from nutrition label features alone. "
        "The sensor (AUC from enzymatic digestion) is added in Phase 3 to beat this baseline."
    ),
    "feature_importance_rf": {
        FEATURES[i]: round(float(importances[i]), 4) for i in feat_order
    },
}

json_path = VAL_DIR / "baseline_results.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results: {json_path}")


print("\n")
print("PHASE 0 SUMMARY")
print("=" * 55)
print(f"  Baseline accuracy (best model):  {best_acc:.1%}")
print(f"  Random chance (3-class):         33.3%")
print(f"  Target with sensor (Phase 3):    75.0%")
print(f"  Gap to close with sensor:        {0.75 - best_acc:.1%}")
