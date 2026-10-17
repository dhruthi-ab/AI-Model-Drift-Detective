"""FastAPI backend.  Run:  uvicorn app:app --reload --port 8000   (from the backend/ folder)"""
import io, os, sys
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(__file__))
from data_gen import make_data
import numpy as np
from datetime import date, datetime

def profile_dataset(df, filename):
    if df.empty or len(df.columns) == 0:
        raise ValueError("The CSV has no data rows or columns.")

    columns = []

    for name in df.columns:
        s = df[name]
        missing = int(s.isna().sum())
        non_missing = s.dropna()
        unique = int(s.nunique())
        numeric = pd.to_numeric(non_missing, errors="coerce")
        is_numeric = len(non_missing) > 0 and numeric.notna().mean() >= 0.9

        item = {
            "name": str(name),
            "type": "numeric" if is_numeric else "categorical",
            "missing_count": missing,
            "missing_pct": round(missing / len(df) * 100, 2),
            "unique_count": unique,
        }

        if is_numeric and len(numeric.dropna()) > 0:
            vals = numeric.dropna()
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1

            item.update({
                "mean": round(float(vals.mean()), 3),
                "median": round(float(vals.median()), 3),
                "min": round(float(vals.min()), 3),
                "max": round(float(vals.max()), 3),
                "outlier_count": int(
                    ((vals < q1 - 1.5 * iqr) |
                     (vals > q3 + 1.5 * iqr)).sum()
                ),
            })
        else:
            item["top_values"] = [
                {"value": str(k), "count": int(v)}
                for k, v in s.astype("string").fillna("(missing)")
                    .value_counts().head(5).items()
            ]

        columns.append(item)

    return {
        "mode": "profile",
        "label": filename,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
    }from service import DriftService

app = FastAPI(title="AI Model Drift Detective")
svc = DriftService()


@app.get("/api/baseline")
def baseline():
    return dict(baseline=svc.cfg["baseline"], features=svc.num + svc.cat, target=svc.target)


@app.post("/api/simulate")
def simulate(drift: float = 0.0, n: int = 2000, seed: int = 7):
    """Generate a production-like batch with a chosen amount of drift (demo mode)."""
    df = make_data(n=n, drift=max(0.0, min(drift, 1.0)), seed=seed)
    return svc.analyze(df, label=f"sim drift={drift:.2f}")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyse an uploaded CSV batch (labels optional)."""
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        return svc.analyze(df, label=file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/retrain")
def retrain():
    try:
        return dict(new_baseline=svc.retrain())
    except ValueError as e:
        raise HTTPException(400, str(e))


# Serve the dashboard at "/"
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend"),
                           html=True), name="ui")
