# AI Model Drift Detective

Monitors a deployed ML model (customer-churn classifier) and detects when it silently degrades.

## Run
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python backend/train.py                              # trains baseline -> artifacts/
cd backend && uvicorn app:app --reload --port 8000
# open http://localhost:8000
```

## Use your own data (e.g. the IBM Telco Customer Churn CSV)
```bash
python backend/train.py --csv path/to/telco.csv --target Churn   # target must be 0/1
```
Then upload later batches (same columns) with the "Upload CSV batch" button.

## Structure
- backend/data_gen.py  synthetic data + controllable drift
- backend/train.py     trains the baseline pipeline, saves model + reference data + baseline metrics
- backend/drift.py     PSI, KS test, chi-square, overall status logic
- backend/service.py   loads artifacts, analyses batches, retrains
- backend/app.py       FastAPI endpoints (/api/simulate, /api/analyze, /api/retrain, /api/baseline)
- frontend/index.html  dashboard (Chart.js)

API docs auto-generated at http://localhost:8000/docs
