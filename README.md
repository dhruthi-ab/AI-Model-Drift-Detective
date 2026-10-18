# AI Model Drift Detective 🔍📊

> **Problem Statement**: Monitors a deployed ML model (customer-churn classifier) and detects when it silently degrades.

A production-ready full-stack MLOps web application designed to monitor live inference streams, detect **covariate drift** and **silent concept degradation**, alert ML engineers, and enable **one-click model remediation & retraining**.

---

## 🌟 Key Features

1. **Authentication & Role-Based Access Control**:
   - Secure sign-in & user registration with PBKDF2-HMAC-SHA256 password hashing.
   - Built-in 1-click quick credentials:
     - 👑 **Admin**: `admin` / `admin123`
     - 🔍 **Analyst**: `analyst` / `analyst123`
   - Session tokens and protected remediation endpoints.

2. **Customer Churn Classifier Model**:
   - Trained on realistic telecommunication customer profiles (`tenure`, `monthly_charges`, `contract_type`, `internet_service`, etc.).
   - Model versioning (`v1.0.0`, `v1.1.0`), tracking accuracy, precision, recall, F1-score, and ROC-AUC.

3. **Multi-Faceted Drift Detection Engine**:
   - **Population Stability Index (PSI)**: Quantifies shift in feature distributions and output probabilities.
   - **Two-Sample Kolmogorov-Smirnov (KS) Test**: Evaluates statistical significance ($p$-values) of continuous numerical shifts.
   - **Silent Concept Degradation Detector**: Measures performance divergence between expected baseline accuracy and live production labels.
   - **Categorical Frequency Shift**: Detects contract type and service tier shifts.

4. **Live Drift Simulator**:
   - **Normal Batch**: Safe baseline telemetry.
   - **Covariate Shift**: Demographic change (younger accounts, tenure drops, fiber charges rise).
   - **Concept Drift**: Competitor buyout campaign causes loyal 2-year accounts to churn unexpectedly without obvious feature anomalies.
   - **Catastrophic Outage**: Widespread shock and mass attrition.
   - **Single Customer Churn Predictor**: Real-time customer inference with out-of-distribution outlier warnings.

5. **Incidents & 1-Click Remediation**:
   - Automated triage of critical and warning drift alerts.
   - 1-Click model retraining and baseline re-calibration.

6. **SQLite Database Explorer**:
   - Real-time viewer of the persistent SQLite database (`data/drift_detective.db`).
   - Paginated customer inference logs, search by Customer ID, and audit history.

---

## 🚀 Quick Start

### 1. Run the Server
From the project folder:
```powershell
cd C:\Users\hp\.gemini\antigravity\scratch\ai-model-drift-detective
python run_server.py
```
Or directly with uvicorn:
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 2. Access the Application
Open your web browser at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📁 Project Structure

```
ai-model-drift-detective/
├── app/
│   ├── database.py         # SQLite database schema, connections & auth hashing
│   ├── model_engine.py     # Customer Churn RandomForest model & data generator
│   ├── drift_detector.py   # PSI, KS-test, and performance degradation algorithms
│   ├── main.py             # FastAPI REST API & static file routes
│   └── static/
│       └── index.html      # Responsive frontend SPA (Tailwind CSS + Chart.js)
├── data/
│   ├── drift_detective.db  # SQLite database storing users, batches, alerts & inferences
│   ├── churn_model.pkl     # Serialized scikit-learn model
│   └── baseline_data.pkl   # Serialized baseline training distribution
├── run_server.py           # Application launcher
└── README.md               # Documentation
```
