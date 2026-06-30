import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "food_gi_dataset_v2.csv"
FIG_DIR   = BASE_DIR / "results" / "figures"
MODEL_DIR = BASE_DIR / "results" / "models"
VAL_DIR   = BASE_DIR / "results" / "validation"

for d in [FIG_DIR, MODEL_DIR, VAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# GI reference values from Foster-Powell et al. 2002 (International Tables
# of Glycemic Index and Glycemic Load Values), n=79 foods.
# This dataset is used here to establish a macronutrient-only baseline —
# i.e., how well nutrition label features alone predict GI category.
# This is CONTEXT for the main project, not the primary result.
# The primary result is Spearman correlation between sensor AUC and
# published GI values across 5 deliberately-spread foods (Phase 2).
df = pd.read_csv(DATA_PATH)
print(f" GI distribution: {dict(df['gi_category'].value_counts())}")

# Macronutrient features available from a standard nutrition label (baseline)
FEATURES = ["carb_g", "fiber_g", "sugar_g", "fat_g", "protein_g"]
TARGET   = "gi_category"
CLASSES  = ["low", "medium", "high"]

X = df[FEATURES].values
y = df[TARGET].values

# PRIMARY ANALYSIS: Spearman correlation between macronutrients and GI value.
# This directly parallels the sensor analysis in Phase 2, where the primary result will be Spearman rho between normalized AUC and published GI value.
# Using rho here (not Pearson) because GI values are not normally distributed and relationship with other macro/micro nutrients may be non-linear

print("\nPRIMARY: Spearman Correlation — Macronutrients vs Published GI Value")
print("=" * 60)

gi_values = df["gi_value"].values  # continuous published GI, not category

spearman_results = {}
for feat in FEATURES:
    rho, pval = stats.spearmanr(df[feat].values, gi_values)
    spearman_results[feat] = {"rho": round(float(rho), 4), "p_value": round(float(pval), 4)}
    sig = "*" if pval < 0.05 else " "
    print(f" {feat:15s}  rho = {rho:+.3f}   p = {pval:.3f} {sig}")

# Plot Spearman correlations
fig, ax = plt.subplots(figsize=(7, 4))
rhos   = [spearman_results[f]["rho"] for f in FEATURES]
colors = ["#1A5276" if r < 0 else "#E74C3C" for r in rhos]
bars = ax.bar(FEATURES, rhos, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Spearman Correlation: Macronutrients vs Published GI Value\n(Baseline — no sensor)", fontsize=11)
ax.set_ylabel("Spearman ρ")
ax.set_ylim(-1, 1)
for bar, r in zip(bars, rhos):
    ax.text(bar.get_x() + bar.get_width() / 2,
            r + (0.03 if r >= 0 else -0.06),
            f"{r:+.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
spearman_fig_path = FIG_DIR / "baseline_spearman_macronutrients.png"
plt.savefig(spearman_fig_path, dpi=150)
plt.close()
print(f"\nSaved: {spearman_fig_path}")

# Context: Classification accuracy (macronutrients → low/medium/high GI).
# This answers a different question than the primary analysis: "Can label data alone classify GI category?"
# Leave-one-out CV is used because n=79 is too small for k-fold to give accurate estimates


print("Classification Accuracy: Macronutrients Only")

models = {
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

loo = LeaveOneOut()
cv_results = {}
best_name, best_acc = None, 0.0

for name, model in models.items():
    acc_scores = cross_val_score(model, X, y, cv=loo, scoring="accuracy")
    f1_scores  = cross_val_score(model, X, y, cv=loo, scoring="f1_weighted")
    acc_mean   = acc_scores.mean()
    f1_mean    = f1_scores.mean()
    cv_results[name] = {
        "loo_accuracy":    round(acc_mean, 4),
        "loo_f1_weighted": round(f1_mean, 4),
    }
    print(f"{name}:")
    print(f" LOO Accuracy:      {acc_mean:.3f} ± {acc_scores.std():.3f}")
    print(f" LOO F1 (weighted): {f1_mean:.3f}")
    if acc_mean > best_acc:
        best_acc, best_name = acc_mean, name

print(f"Best model: {best_name} ({best_acc:.3f})")
print(f"Random chance (3-class): 33.3%") #did i do the math correct? 

# Confusion matrix for best model
best_model = models[best_name]
y_pred = cross_val_predict(best_model, X, y, cv=loo)
cm = confusion_matrix(y, y_pred, labels=CLASSES)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(
    f"Baseline Confusion Matrix (Context Only)\n"
    f"{best_name} — LOO CV, n={len(df)} foods",
    fontsize=11, pad=10
)
plt.tight_layout()
cm_path = FIG_DIR / "baseline_confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\nSaved: {cm_path}")

# Feature importance (Random Forest fit on all data — for exploratory context only,
# not cross-validated; treat rankings as approximate given small n).
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
feat_order  = np.argsort(importances)[::-1]

print("\nFeature Importance (Random Forest, exploratory — not cross-validated):")
for i in feat_order:
    bar = "█" * int(importances[i] * 40)
    print(f" {FEATURES[i]:15s} {importances[i]:.3f} {bar}")

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    [FEATURES[i] for i in feat_order],
    [importances[i] for i in feat_order],
    color=["#4A90D9"] * len(FEATURES), edgecolor="white", linewidth=0.5
)
ax.set_title("Feature Importance — Baseline RF (macronutrients only)\nExploratory, not cross-validated", fontsize=11)
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

# Save model
model_path = MODEL_DIR / "baseline_rf_model.joblib"
joblib.dump(rf, model_path)
print(f"Saved model: {model_path}")

# Save results JSON
results = {
    "description": (
        "Macronutrient-only baseline using Foster-Powell 2002 GI reference set. "
        "This is context for the main project, not the primary result. "
        "Primary result is Spearman correlation between sensor AUC and published "
        "GI values across 5 foods (Phase 2)."
    ),
    "n_foods": len(df),
    "reference": "Foster-Powell et al. 2002, International Tables of GI and GL Values",
    "class_distribution": {k: int(v) for k, v in df["gi_category"].value_counts().items()},
    "features_used": FEATURES,
    "primary_analysis": {
        "method": "Spearman correlation (macronutrients vs continuous GI value)",
        "results": spearman_results,
        "interpretation": (
            "Shows which macronutrients rank-correlate with GI. "
            "Fiber expected negative (slows digestion), carbs/sugar expected positive."
        ),
    },
    "context_classification": {
        "method": "Leave-One-Out cross-validation (3-class: low/medium/high GI)",
        "models": cv_results,
        "best_model": best_name,
        "best_loo_accuracy": round(best_acc, 4),
        "random_chance_accuracy": 0.333,
        "interpretation": (
            "60.8% LOO accuracy confirms nutrition label data alone is insufficient "
            "for reliable GI classification, motivating the physical sensor."
        ),
    },
    "feature_importance_rf": {
        FEATURES[i]: round(float(importances[i]), 4) for i in feat_order
    },
}

json_path = VAL_DIR / "baseline_results.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results: {json_path}")

print("BASELINE SUMMARY")
print(f" n foods: {len(df)}")
best_feat = max(spearman_results, key=lambda f: abs(spearman_results[f]['rho']))
print(f" Best Spearman rho ({best_feat}):  {spearman_results[best_feat]['rho']:+.3f}")
print(f" Classification accuracy:    {best_acc:.1%} (best model, LOO CV)")
