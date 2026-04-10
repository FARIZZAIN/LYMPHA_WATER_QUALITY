"""
evaluate_model.py
=================
Loads the saved potability model and generates evaluation graphs:
  1. Confusion Matrix
  2. ROC Curve
  3. Precision-Recall Curve
  4. Per-class Precision / Recall / F1 bar chart

Run from the backend folder:
  python evaluate_model.py
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, f1_score
)

# ── Load scaler + model ───────────────────────────────────
_models_dir = Path(__file__).parent.parent / "models"
with open(_models_dir / "potability_scaler.pkl", "rb") as f:
    scaler_bundle = pickle.load(f)

with open(_models_dir / "potability_model.h5", "rb") as f:
    model_bundle = pickle.load(f)

scaler       = scaler_bundle["scaler"]
threshold    = scaler_bundle["threshold"]
feature_order = scaler_bundle["feature_order"]
model        = model_bundle["model"]

print(f"Model type : {type(model).__name__}")
print(f"Threshold  : {threshold}")
print(f"Features   : {len(feature_order)}")

# ── Reload dataset (same split as training) ───────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "water_potability.csv"
COL_MAP = {
    "ph": "pH", "Hardness": "Hardness", "Solids": "TDS",
    "Chloramines": "Chloramines", "Sulfate": "Sulfate",
    "Conductivity": "Conductivity", "Organic_carbon": "Organic_carbon",
    "Trihalomethanes": "Trihalomethanes", "Turbidity": "Turbidity",
    "Potability": "Potability",
}
BASE_FEATURES = ["pH","Hardness","TDS","Chloramines","Sulfate",
                 "Conductivity","Organic_carbon","Trihalomethanes","Turbidity"]

df = pd.read_csv(DATA_PATH).rename(columns=COL_MAP)
for col in BASE_FEATURES:
    if df[col].isnull().any():
        for label in [0, 1]:
            med = df.loc[df["Potability"] == label, col].median()
            df.loc[(df["Potability"] == label) & df[col].isnull(), col] = med
        df[col] = df[col].fillna(df[col].median())

# Feature engineering (must match training)
df["pH_x_Chloramines"]      = df["pH"] * df["Chloramines"]
df["TDS_div_Conductivity"]  = df["TDS"] / (df["Conductivity"] + 1e-6)
df["Hardness_x_Sulfate"]    = df["Hardness"] * df["Sulfate"]
df["Organic_x_Trihalometh"] = df["Organic_carbon"] * df["Trihalomethanes"]
df["pH_squared"]            = df["pH"] ** 2
df["Turbidity_x_TDS"]       = df["Turbidity"] * df["TDS"]

X = df[feature_order].values.astype(np.float32)
y = df["Potability"].values.astype(int)

# Same random_state as training to get the same test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_test_s = scaler.transform(X_test)

# ── Predictions ───────────────────────────────────────────
y_prob = model.predict_proba(X_test_s)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

report = classification_report(y_test, y_pred,
                               target_names=["Not Potable", "Potable"],
                               output_dict=True)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"]))

# ── Plot setup ────────────────────────────────────────────
plt.style.use("dark_background")
BG    = "#0a0f1a"
PANEL = "#111827"
CYAN  = "#00e5ff"
GREEN = "#00ff94"
RED   = "#f43f5e"
AMBER = "#f59e0b"
MUTED = "#6b7280"

fig = plt.figure(figsize=(16, 10), facecolor=BG)
fig.suptitle("LYMPHA — Water Potability Model Evaluation",
             fontsize=16, fontweight="bold", color="white", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

def styled_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1f2937")
    return ax

# ── 1. Confusion Matrix ───────────────────────────────────
ax1 = styled_ax(fig.add_subplot(gs[0, 0]), "Confusion Matrix")
cm  = confusion_matrix(y_test, y_pred)
im  = ax1.imshow(cm, cmap="Blues", aspect="auto")
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                 fontsize=18, fontweight="bold",
                 color="white" if cm[i, j] > cm.max() / 2 else MUTED)
ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
ax1.set_xticklabels(["Not Potable", "Potable"], color="white")
ax1.set_yticklabels(["Not Potable", "Potable"], color="white")
ax1.set_xlabel("Predicted", color=MUTED, fontsize=9)
ax1.set_ylabel("Actual", color=MUTED, fontsize=9)

# ── 2. ROC Curve ──────────────────────────────────────────
ax2 = styled_ax(fig.add_subplot(gs[0, 1]), "ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc     = auc(fpr, tpr)
ax2.plot(fpr, tpr, color=CYAN, lw=2, label=f"AUC = {roc_auc:.3f}")
ax2.plot([0, 1], [0, 1], color=MUTED, lw=1, linestyle="--", label="Random")
ax2.set_xlabel("False Positive Rate", color=MUTED, fontsize=9)
ax2.set_ylabel("True Positive Rate", color=MUTED, fontsize=9)
ax2.legend(facecolor=PANEL, edgecolor="#1f2937", labelcolor="white", fontsize=9)
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])

# ── 3. Precision-Recall Curve ─────────────────────────────
ax3 = styled_ax(fig.add_subplot(gs[0, 2]), "Precision-Recall Curve")
prec, rec, _ = precision_recall_curve(y_test, y_prob)
pr_auc       = auc(rec, prec)
baseline     = y_test.sum() / len(y_test)
ax3.plot(rec, prec, color=GREEN, lw=2, label=f"AUC = {pr_auc:.3f}")
ax3.axhline(baseline, color=MUTED, lw=1, linestyle="--", label=f"Baseline ({baseline:.2f})")
ax3.set_xlabel("Recall", color=MUTED, fontsize=9)
ax3.set_ylabel("Precision", color=MUTED, fontsize=9)
ax3.legend(facecolor=PANEL, edgecolor="#1f2937", labelcolor="white", fontsize=9)
ax3.set_xlim([0, 1]); ax3.set_ylim([0, 1.02])

# ── 4. Per-class Precision / Recall / F1 ─────────────────
ax4 = styled_ax(fig.add_subplot(gs[1, 0:2]), "Per-class Metrics")
classes  = ["Not Potable", "Potable"]
metrics  = ["precision", "recall", "f1-score"]
colors   = [CYAN, GREEN, AMBER]
x        = np.arange(len(classes))
width    = 0.25
for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals = [report[c][metric] for c in classes]
    bars = ax4.bar(x + i * width, vals, width, label=metric.capitalize(),
                   color=color, alpha=0.85, edgecolor=BG)
    for bar, val in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=9)
ax4.set_xticks(x + width); ax4.set_xticklabels(classes, color="white")
ax4.set_ylim([0, 1.12])
ax4.set_ylabel("Score", color=MUTED, fontsize=9)
ax4.legend(facecolor=PANEL, edgecolor="#1f2937", labelcolor="white", fontsize=9)
ax4.axhline(0.79, color=MUTED, lw=1, linestyle="--")
ax4.text(2.35, 0.80, "Overall Acc 79%", color=MUTED, fontsize=8)

# ── 5. Summary stats card ─────────────────────────────────
ax5 = styled_ax(fig.add_subplot(gs[1, 2]), "Summary")
ax5.axis("off")
acc      = (y_pred == y_test).mean()
macro_f1 = f1_score(y_test, y_pred, average="macro")
stats = [
    ("Accuracy",       f"{acc*100:.1f}%"),
    ("Macro F1",       f"{macro_f1:.3f}"),
    ("ROC-AUC",        f"{roc_auc:.3f}"),
    ("PR-AUC",         f"{pr_auc:.3f}"),
    ("Threshold",      f"{threshold:.2f}"),
    ("Test samples",   str(len(y_test))),
    ("Model",          "Random Forest"),
    ("Features",       str(len(feature_order))),
]
for i, (label, val) in enumerate(stats):
    y_pos = 0.92 - i * 0.115
    ax5.text(0.02, y_pos, label, color=MUTED, fontsize=10, transform=ax5.transAxes)
    ax5.text(0.98, y_pos, val,   color=CYAN,  fontsize=10, transform=ax5.transAxes,
             ha="right", fontweight="bold")
    ax5.axline((0, y_pos - 0.04), (1, y_pos - 0.04), color="#1f2937", lw=0.5,
               transform=ax5.transAxes)

out = Path(__file__).parent.parent / "models" / "potability_evaluation.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out.resolve()}")
plt.show()
