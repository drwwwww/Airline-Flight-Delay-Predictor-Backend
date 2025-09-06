
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"             
FEATURES_PATH = HERE / "features.json"
MODEL_PATH = HERE / "model.pkl"


feat = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
numeric = feat["numeric"]            #["depHour"]
categorical = feat["categorical"]    #["origin","dest"]


csvs = sorted(DATA_DIR.glob("*.csv"))
if not csvs:
    raise SystemExit(f"No CSVs found in {DATA_DIR}")
df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)


if "origin" in categorical: df["origin"] = df["origin"].astype(str).str.strip().str.upper()
if "dest"   in categorical: df["dest"]   = df["dest"].astype(str).str.strip().str.upper()

need = set(numeric + categorical + ["depDelayMinutes"])
df = df.dropna(subset=list(need))

X = df[numeric + categorical].copy()
y = (df["depDelayMinutes"] >= 15).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pre = ColumnTransformer([
    ("num", "passthrough", numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])
clf = LogisticRegression(max_iter=500, class_weight="balanced")
pipe = Pipeline([("pre", pre), ("clf", clf)])

pipe.fit(X_train, y_train)
p_test = pipe.predict_proba(X_test)[:, 1]
print(f"AUC (holdout): {roc_auc_score(y_test, p_test):.3f}")

joblib.dump(pipe, MODEL_PATH.open("wb"))
print("Saved:", MODEL_PATH.resolve())
