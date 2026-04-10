"""
train_fish_model.py  —  Fish Species Survival Classifier
=========================================================
Generates realistic synthetic training data and trains a
multi-label GradientBoosting classifier to predict which
fish species can survive in given water conditions.

Features used (5 total):
  Live from sensors (3):  pH, Temperature, Turbidity
  Simulated safely (2):   DO (dissolved oxygen), Conductivity

Species (6):
  Goldfish, Tilapia, Guppy, Mrigal, Silver Carp, Koi Carp

Output files (save next to main.py):
  fish_model.pkl   — trained MultiOutputClassifier
  fish_scaler.pkl  — fitted StandardScaler
  fish_mlb.pkl     — fitted MultiLabelBinarizer (maps indices → species names)

Usage:
  python train_fish_model.py
"""

import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, f1_score

# ─────────────────────────────────────────────────────
# 1. SPECIES SURVIVAL THRESHOLDS
#    Based on real aquaculture / limnology literature.
#    Each species has a min/max for each parameter.
#    A species "survives" if ALL of its conditions are met.
# ─────────────────────────────────────────────────────
SPECIES_CONDITIONS = {
    "Goldfish": {
        "pH":          (6.5, 8.0),
        "Temperature": (10.0, 24.0),   # cold-water fish
        "Turbidity":   (0.0, 20.0),
        "DO":          (5.0, 15.0),
        "Conductivity":(100, 500),
    },
    "Tilapia": {
        "pH":          (6.0, 9.0),
        "Temperature": (22.0, 35.0),   # warm-water fish
        "Turbidity":   (0.0, 30.0),
        "DO":          (3.0, 15.0),    # more tolerant of low DO
        "Conductivity":(100, 1000),
    },
    "Guppy": {
        "pH":          (6.5, 8.5),
        "Temperature": (18.0, 28.0),
        "Turbidity":   (0.0, 15.0),
        "DO":          (5.0, 15.0),
        "Conductivity":(100, 600),
    },
    "Mrigal": {
        "pH":          (6.5, 8.5),
        "Temperature": (18.0, 32.0),
        "Turbidity":   (0.0, 25.0),
        "DO":          (4.0, 15.0),
        "Conductivity":(100, 700),
    },
    "Silver Carp": {
        "pH":          (6.5, 8.5),
        "Temperature": (18.0, 32.0),
        "Turbidity":   (0.0, 30.0),
        "DO":          (3.0, 15.0),
        "Conductivity":(100, 800),
    },
    "Koi Carp": {
        "pH":          (6.5, 8.0),
        "Temperature": (10.0, 28.0),
        "Turbidity":   (0.0, 20.0),
        "DO":          (5.0, 15.0),
        "Conductivity":(100, 1200),
    },
}

FEATURES    = ["pH", "Temperature", "Turbidity", "DO", "Conductivity"]
SPECIES     = list(SPECIES_CONDITIONS.keys())

# Full realistic ranges for data generation
PARAM_RANGES = {
    "pH":          (5.5, 10.0),
    "Temperature": (8.0,  38.0),
    "Turbidity":   (0.0,  40.0),
    "DO":          (1.0,  15.0),
    "Conductivity":(50,   1400),
}


# ─────────────────────────────────────────────────────
# 2. LABEL FUNCTION
# ─────────────────────────────────────────────────────
def get_surviving_species(sample: dict) -> list:
    surviving = []
    for species, conditions in SPECIES_CONDITIONS.items():
        survives = all(
            conditions[feat][0] <= sample[feat] <= conditions[feat][1]
            for feat in FEATURES
        )
        if survives:
            surviving.append(species)
    return surviving   # empty list = no species survives


# ─────────────────────────────────────────────────────
# 3. GENERATE TRAINING DATA
#    We generate a large, balanced dataset that covers
#    all parts of the parameter space — not just the
#    narrow "safe" region the original CSV was stuck in.
# ─────────────────────────────────────────────────────
print("Generating training data...")

rng = np.random.default_rng(seed=42)
N   = 6000   # samples — enough for a 6-class multi-label problem

rows, labels = [], []

for _ in range(N):
    sample = {feat: float(rng.uniform(*PARAM_RANGES[feat])) for feat in FEATURES}
    surviving = get_surviving_species(sample)
    rows.append([sample[f] for f in FEATURES])
    labels.append(surviving)

X = np.array(rows)
raw_labels = labels

# Remove samples where NO species survives (not useful for training)
valid = [i for i, l in enumerate(raw_labels) if len(l) > 0]
X      = X[valid]
labels = [raw_labels[i] for i in valid]

print(f"  Total generated : {N}")
print(f"  Valid samples   : {len(X)} (at least 1 species survives)")

# Species frequency
from collections import Counter
species_counts = Counter(s for l in labels for s in l)
print("\n  Species frequency:")
for sp in SPECIES:
    c = species_counts.get(sp, 0)
    print(f"    {sp:15s}: {c:4d}  ({100*c/len(X):.1f}%)")

# Save generated training data to CSV
df = pd.DataFrame(X, columns=FEATURES)
df["Survivable_Fish"] = [", ".join(l) for l in labels]
df.to_csv(Path(__file__).parent / "fish_training_data.csv", index=False)
print(f"\n  Saved {len(df)} rows to fish_training_data.csv")


# ─────────────────────────────────────────────────────
# 4. ENCODE LABELS
# ─────────────────────────────────────────────────────
mlb = MultiLabelBinarizer(classes=SPECIES)
y   = mlb.fit_transform(labels)


# ─────────────────────────────────────────────────────
# 5. SCALE FEATURES
# ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)


# ─────────────────────────────────────────────────────
# 6. TRAIN MODEL
#    GradientBoosting per species — fast, accurate,
#    handles class imbalance well.
# ─────────────────────────────────────────────────────
print("\nTraining model...")

base = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.85,
    random_state=42,
)
model = MultiOutputClassifier(base, n_jobs=-1)
model.fit(X_train_sc, y_train)
print("  Done.")


# ─────────────────────────────────────────────────────
# 7. EVALUATE
# ─────────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)

hamming_acc = 1 - hamming_loss(y_test, y_pred)
micro_f1    = f1_score(y_test, y_pred, average="micro", zero_division=0)
macro_f1    = f1_score(y_test, y_pred, average="macro", zero_division=0)

print(f"\n{'─'*40}")
print(f"  Hamming accuracy : {hamming_acc*100:.2f}%")
print(f"  Micro F1         : {micro_f1*100:.2f}%")
print(f"  Macro F1         : {macro_f1*100:.2f}%")
print(f"{'─'*40}")

# Per-species accuracy
print("\n  Per-species accuracy:")
for i, sp in enumerate(mlb.classes_):
    col_acc = (y_test[:, i] == y_pred[:, i]).mean()
    print(f"    {sp:15s}: {col_acc*100:.1f}%")


# ─────────────────────────────────────────────────────
# 8. SAVE
# ─────────────────────────────────────────────────────
print("\nSaving files...")
_models_dir = Path(__file__).parent.parent / "models"
_models_dir.mkdir(exist_ok=True)

with open(_models_dir / "fish_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("  fish_model.pkl   saved")

with open(_models_dir / "fish_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("  fish_scaler.pkl  saved")

with open(_models_dir / "fish_mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)
print("  fish_mlb.pkl     saved")

print("\nDone! Run main.py to serve predictions via /predict/fish")
print(f"Features expected: {FEATURES}")
print(f"Species predicted: {list(mlb.classes_)}")
