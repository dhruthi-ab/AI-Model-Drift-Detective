"""FastAPI backend.  Run:  uvicorn app:app --reload --port 8000   (from the backend/ folder)"""
import io, os, sys
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(__file__))
from data_gen import make_data
from service import DriftService

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
