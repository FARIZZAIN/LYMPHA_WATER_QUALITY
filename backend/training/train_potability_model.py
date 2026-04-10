"""
train_potability_model.py
=========================
Trains the water potability model on the Kaggle Water Potability dataset.

Dataset: https://www.kaggle.com/datasets/adityakadiwal/water-potability
File expected at: backend/data/water_potability.csv

Physical sensors (ESP32):  pH, Solids (TDS), Conductivity, Turbidity
Simulated at runtime:      Hardness, Chloramines, Sulfate, Organic_carbon, Trihalomethanes

Tries: Neural Net, Random Forest, Gradient Boosting, XGBoost, Stacking Ensemble.
Applies SMOTE for class balancing and feature engineering for interaction terms.
Saves whichever scores highest on balanced macro-F1.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed - skipping. Run: pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[WARN] imbalanced-learn not installed - skipping SMOTE. Run: pip install imbalanced-learn")

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "data" / "water_potability.csv"

COL_MAP = {
    "ph":               "pH",
    "Hardness":         "Hardness",
    "Solids":           "TDS",
    "Chloramines":      "Chloramines",
    "Sulfate":          "Sulfate",
    "Conductivity":     "Conductivity",
    "Organic_carbon":   "Organic_carbon",
    "Trihalomethanes":  "Trihalomethanes",
    "Turbidity":        "Turbidity",
    "Potability":       "Potability",
}

BASE_FEATURES = [
    "pH", "Hardness", "TDS", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]

# ---------------------------------------------
# LOAD + IMPUTE
# ---------------------------------------------
print(f"Loading {DATA_PATH} ...")
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\n[ERROR] Dataset not found at {DATA_PATH}\n"
        "Download from: https://www.kaggle.com/datasets/adityakadiwal/water-potability\n"
        "Place water_potability.csv in: backend/data/"
    )

df = pd.read_csv(DATA_PATH).rename(columns=COL_MAP)
print(f"  {len(df)} rows  |  class balance: {df['Potability'].value_counts().to_dict()}")

for col in BASE_FEATURES:
    if df[col].isnull().any():
        for label in [0, 1]:
            med = df.loc[df["Potability"] == label, col].median()
            df.loc[(df["Potability"] == label) & df[col].isnull(), col] = med
        df[col] = df[col].fillna(df[col].median())

# ---------------------------------------------
# FEATURE ENGINEERING
# Interaction and ratio terms that domain knowledge suggests matter
# ---------------------------------------------
print("Engineering features...")
df["pH_x_Chloramines"]      = df["pH"] * df["Chloramines"]
df["TDS_div_Conductivity"]  = df["TDS"] / (df["Conductivity"] + 1e-6)
df["Hardness_x_Sulfate"]    = df["Hardness"] * df["Sulfate"]
df["Organic_x_Trihalometh"] = df["Organic_carbon"] * df["Trihalomethanes"]
df["pH_squared"]            = df["pH"] ** 2
df["Turbidity_x_TDS"]       = df["Turbidity"] * df["TDS"]

ENGINEERED = [
    "pH_x_Chloramines", "TDS_div_Conductivity", "Hardness_x_Sulfate",
    "Organic_x_Trihalometh", "pH_squared", "Turbidity_x_TDS",
]
FEATURE_ORDER = BASE_FEATURES + ENGINEERED
print(f"  Total features: {len(FEATURE_ORDER)}")

# ---------------------------------------------
# SPLIT
# ---------------------------------------------
X = df[FEATURE_ORDER].values.astype(np.float32)
y = df["Potability"].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)
print(f"  train:{len(X_train)}  val:{len(X_val)}  test:{len(X_test)}")

# ---------------------------------------------
# SCALE
# ---------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# Class weights
cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: cw[0], 1: cw[1]}

# ---------------------------------------------
# SMOTE - oversample minority class in training set only
# ---------------------------------------------
if HAS_SMOTE:
    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
    print(f"  After SMOTE - train size: {len(X_train_s)}, balance: {np.bincount(y_train.astype(int))}")

results = {}

# ---------------------------------------------
# MODEL 1 - Random Forest
# ---------------------------------------------
print("\n-- Random Forest -----------------------")
rf = RandomForestClassifier(
    n_estimators=500, max_depth=None, min_samples_leaf=2,
    max_features="sqrt", class_weight="balanced_subsample",
    random_state=42, n_jobs=-1
)
rf.fit(X_train_s, y_train)
rf_preds = rf.predict(X_test_s)
rf_f1    = f1_score(y_test, rf_preds, average="macro")
print(f"  macro-F1: {rf_f1:.4f}")
print(classification_report(y_test, rf_preds, target_names=["Not Potable", "Potable"]))
results["RF"] = (rf_f1, rf, 0.5, "rf")

# ---------------------------------------------
# MODEL 2 - Extra Trees (faster, often similar to RF)
# ---------------------------------------------
print("\n-- Extra Trees -------------------------")
et = ExtraTreesClassifier(
    n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
    random_state=42, n_jobs=-1
)
et.fit(X_train_s, y_train)
et_preds = et.predict(X_test_s)
et_f1    = f1_score(y_test, et_preds, average="macro")
print(f"  macro-F1: {et_f1:.4f}")
print(classification_report(y_test, et_preds, target_names=["Not Potable", "Potable"]))
results["ET"] = (et_f1, et, 0.5, "rf")  # same predict interface as RF

# ---------------------------------------------
# MODEL 3 - Gradient Boosting
# ---------------------------------------------
print("\n-- Gradient Boosting -------------------")
sw = np.where(y_train == 1, cw[1], cw[0])
gb = GradientBoostingClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=5,
    subsample=0.8, min_samples_leaf=10, random_state=42
)
gb.fit(X_train_s, y_train, sample_weight=sw)

gb_val_probs = gb.predict_proba(X_val_s)[:, 1]
best_t, best_f = 0.5, 0.0
for t in np.arange(0.30, 0.71, 0.01):
    f = f1_score(y_val, (gb_val_probs >= t).astype(int), average="macro")
    if f > best_f: best_f, best_t = f, t
gb_preds = (gb.predict_proba(X_test_s)[:, 1] >= best_t).astype(int)
gb_f1    = f1_score(y_test, gb_preds, average="macro")
print(f"  threshold:{best_t:.2f}  macro-F1: {gb_f1:.4f}")
print(classification_report(y_test, gb_preds, target_names=["Not Potable", "Potable"]))
results["GB"] = (gb_f1, gb, best_t, "gb")

# ---------------------------------------------
# MODEL 4 - XGBoost
# ---------------------------------------------
if HAS_XGB:
    print("\n-- XGBoost -----------------------------")
    scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss", random_state=42,
        n_jobs=-1, verbosity=0
    )
    xgb.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
            verbose=False)

    xgb_val_probs = xgb.predict_proba(X_val_s)[:, 1]
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.30, 0.71, 0.01):
        f = f1_score(y_val, (xgb_val_probs >= t).astype(int), average="macro")
        if f > best_f: best_f, best_t = f, t
    xgb_preds = (xgb.predict_proba(X_test_s)[:, 1] >= best_t).astype(int)
    xgb_f1    = f1_score(y_test, xgb_preds, average="macro")
    print(f"  threshold:{best_t:.2f}  macro-F1: {xgb_f1:.4f}")
    print(classification_report(y_test, xgb_preds, target_names=["Not Potable", "Potable"]))
    results["XGB"] = (xgb_f1, xgb, best_t, "gb")

# ---------------------------------------------
# MODEL 5 - Stacking Ensemble
# ---------------------------------------------
print("\n-- Stacking Ensemble -------------------")
base_estimators = [
    ("rf",  RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
    ("et",  ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=43, n_jobs=-1)),
    ("gb",  GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=44)),
]
if HAS_XGB:
    base_estimators.append(
        ("xgb", XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                               scale_pos_weight=scale_pos, random_state=45, verbosity=0, n_jobs=-1))
    )

stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
    cv=5, n_jobs=-1, passthrough=True
)
stack.fit(X_train_s, y_train)

stack_val_probs = stack.predict_proba(X_val_s)[:, 1]
best_t, best_f = 0.5, 0.0
for t in np.arange(0.30, 0.71, 0.01):
    f = f1_score(y_val, (stack_val_probs >= t).astype(int), average="macro")
    if f > best_f: best_f, best_t = f, t
stack_preds = (stack.predict_proba(X_test_s)[:, 1] >= best_t).astype(int)
stack_f1    = f1_score(y_test, stack_preds, average="macro")
print(f"  threshold:{best_t:.2f}  macro-F1: {stack_f1:.4f}")
print(classification_report(y_test, stack_preds, target_names=["Not Potable", "Potable"]))
results["STACK"] = (stack_f1, stack, best_t, "gb")

# ---------------------------------------------
# PICK BEST
# ---------------------------------------------
best_key = max(results, key=lambda k: results[k][0])
best_f1_val, best_model, best_threshold, best_type = results[best_key]

print(f"\n{'='*48}")
print(f"  Results summary:")
for k, (f, _, t, _) in sorted(results.items(), key=lambda x: -x[1][0]):
    marker = " <-- BEST" if k == best_key else ""
    print(f"    {k:8s}  macro-F1={f:.4f}  threshold={t:.2f}{marker}")
print(f"{'='*48}")

# ---------------------------------------------
# SAVE
# ---------------------------------------------
print("\nSaving...")
_models_dir = Path(__file__).parent.parent / "models"
_models_dir.mkdir(exist_ok=True)
with open(_models_dir / "potability_model.h5", "wb") as f:
    pickle.dump({"model": best_model, "type": best_type, "threshold": best_threshold}, f)
print(f"  potability_model.h5  saved  ({best_key})")

with open(_models_dir / "potability_scaler.pkl", "wb") as f:
    pickle.dump({
        "scaler":       scaler,
        "threshold":    best_threshold,
        "model_type":   best_type,
        "feature_order": FEATURE_ORDER,
    }, f)
print("  potability_scaler.pkl  saved")
print(f"\nBest: {best_key}  macro-F1={best_f1_val:.4f}  threshold={best_threshold:.2f}")
print("Restart main.py to use the new model.")
