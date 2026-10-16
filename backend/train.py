"""Train the baseline model and save everything the detector needs.

Usage:
    python backend/train.py                      # synthetic data
    python backend/train.py --csv data.csv --target Churn   # your own data
"""
import argparse, json, os, sys
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from data_gen import make_data, NUM_COLS, CAT_COLS, TARGET

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def build_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    return Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=42))])


def metrics(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return dict(accuracy=round(accuracy_score(y, p > .5), 4), f1=round(f1_score(y, p > .5), 4),
                auc=round(roc_auc_score(y, p), 4))


def train(df, num_cols, cat_cols, target, save=True):
    X, y = df[num_cols + cat_cols], df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = build_pipeline(num_cols, cat_cols).fit(Xtr, ytr)
    base = metrics(model, Xte, yte)
    config = dict(num_cols=num_cols, cat_cols=cat_cols, target=target, baseline=base,
                  n_train=len(Xtr))
    if save:
        os.makedirs(ART, exist_ok=True)
        joblib.dump(model, os.path.join(ART, "model.joblib"))
        Xtr.assign(**{target: ytr}).to_csv(os.path.join(ART, "reference.csv"), index=False)  # training snapshot = "normal"
        json.dump(config, open(os.path.join(ART, "config.json"), "w"), indent=2)
    return model, Xtr, config


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--target", default=TARGET)
    a = ap.parse_args()
    if a.csv:
        df = pd.read_csv(a.csv).dropna()
        cat = [c for c in df.columns if c != a.target and df[c].dtype == object]
        num = [c for c in df.columns if c != a.target and c not in cat]
    else:
        df, num, cat, a.target = make_data(8000, drift=0.0, seed=1), NUM_COLS, CAT_COLS, TARGET
    _, _, cfg = train(df, num, cat, a.target)
    print("Baseline metrics:", cfg["baseline"])
