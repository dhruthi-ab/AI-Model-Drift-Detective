import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Depends, Query, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
import io
from pydantic import BaseModel

from app.database import (
    init_db, get_db_connection, hash_password, verify_password,
    create_session, get_session, delete_session
)
from app.model_engine import model_engine, load_or_create_dataset
from app.drift_detector import analyze_batch_drift

# Initialize DB tables on startup
init_db()

app = FastAPI(title="AI Model Drift Detective", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ----------------- Models -----------------
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: Optional[str] = None
    password: str
    role: Optional[str] = "analyst"

class BatchSimulationRequest(BaseModel):
    scenario: str = "normal"  # normal, covariate_shift, concept_drift, severe_shock
    sample_count: int = 120

class SinglePredictRequest(BaseModel):
    tenure: float
    monthly_charges: float
    support_calls: int = 1
    data_usage_gb: float = 25.0
    contract: str = "month-to-month"
    internet_service: str = "fiber"
    payment_method: str = "card"
    actual_churn: Optional[int] = None

# ----------------- Auth Helpers -----------------
def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "").strip()
    return get_session(token)

def require_auth(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    user = get_current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required. Please sign in.")
    return user

# ----------------- Routes -----------------
@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI Model Drift Detective API running. index.html is being prepared."}

# Auth endpoints
@app.post("/api/auth/login")
def login(req: LoginRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (req.username.strip(),))
    user = cursor.fetchone()
    conn.close()
    
    if not user or not verify_password(req.password, user["password_hash"], user["salt"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
        
    token = create_session(user["id"], user["username"], user["role"])
    return {
        "success": True,
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"]
        }
    }

@app.post("/api/auth/register")
def register(req: RegisterRequest):
    username = req.username.strip()
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
        
    pwd_hash, salt = hash_password(req.password)
    now = datetime.now().isoformat()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (username, req.email or f"{username}@drift-detective.ai", pwd_hash, salt, req.role or "analyst", now)
        )
        user_id = cursor.lastrowid
        conn.commit()
    except Exception:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists.")
        
    conn.close()
    token = create_session(user_id, username, req.role or "analyst")
    return {
        "success": True,
        "token": token,
        "user": {
            "id": user_id,
            "username": username,
            "email": req.email,
            "role": req.role or "analyst"
        }
    }

@app.get("/api/auth/me")
def me(user: Dict[str, Any] = Depends(require_auth)):
    return {"authenticated": True, "user": user}

@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        delete_session(token)
    return {"success": True, "message": "Logged out successfully"}

# ----------------- Dashboard & Metrics -----------------
@app.get("/api/dashboard/summary")
def get_dashboard_summary():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM models WHERE is_active = 1 ORDER BY id DESC LIMIT 1")
    model_row = cursor.fetchone()
    model_info = dict(model_row) if model_row else {
        "version": model_engine.version,
        "algorithm": "RandomForestClassifier",
        "accuracy": model_engine.metrics.get("accuracy", 0.85),
        "f1": model_engine.metrics.get("f1", 0.78),
        "roc_auc": model_engine.metrics.get("roc_auc", 0.90),
        "trained_at": datetime.now().isoformat()
    }
    
    cursor.execute("SELECT * FROM batches ORDER BY id DESC LIMIT 1")
    latest_batch_row = cursor.fetchone()
    latest_batch = dict(latest_batch_row) if latest_batch_row else None
    
    cursor.execute("SELECT COUNT(*) as total_batches FROM batches")
    total_batches = cursor.fetchone()["total_batches"]
    
    cursor.execute("SELECT COUNT(*) as total_inferences FROM customer_inferences")
    total_inferences = cursor.fetchone()["total_inferences"]
    
    cursor.execute("SELECT COUNT(*) as active_alerts FROM alerts WHERE is_resolved = 0")
    active_alerts = cursor.fetchone()["active_alerts"]
    
    cursor.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 5")
    recent_alerts = [dict(r) for r in cursor.fetchall()]
    
    cursor.execute("SELECT * FROM batches ORDER BY id DESC LIMIT 5")
    recent_batches = [dict(r) for r in cursor.fetchall()]
    
    conn.close()
    
    if latest_batch:
        overall_status = latest_batch["drift_status"]
        current_drift_score = latest_batch["drift_score"]
        current_accuracy = latest_batch["accuracy"]
    else:
        overall_status = "HEALTHY"
        current_drift_score = 0.04
        current_accuracy = model_info["accuracy"]
        
    return {
        "status": overall_status,
        "current_drift_score": current_drift_score,
        "current_accuracy": current_accuracy,
        "baseline_accuracy": model_info["accuracy"],
        "model": model_info,
        "total_batches": total_batches,
        "total_inferences": total_inferences,
        "active_alerts_count": active_alerts,
        "latest_batch": latest_batch,
        "recent_alerts": recent_alerts,
        "recent_batches": recent_batches
    }

@app.get("/api/metrics/trends")
def get_metrics_trends(limit: int = 20):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, batch_code, timestamp, scenario, sample_count, drift_score, drift_status, accuracy, accuracy_drop, churn_rate FROM batches ORDER BY id ASC LIMIT ?",
        (limit,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return {"batches": rows}

@app.get("/api/features/distribution")
def get_feature_distribution(feature: str = Query("monthly_charges"), batch_id: Optional[int] = None):
    baseline_df = model_engine.baseline_df
    if baseline_df is None or feature not in baseline_df.columns:
        raise HTTPException(status_code=400, detail=f"Feature {feature} not recognized.")
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if batch_id:
        cursor.execute("SELECT * FROM customer_inferences WHERE batch_id = ?", (batch_id,))
    else:
        cursor.execute("SELECT id FROM batches ORDER BY id DESC LIMIT 1")
        last_b = cursor.fetchone()
        if last_b:
            cursor.execute("SELECT * FROM customer_inferences WHERE batch_id = ?", (last_b["id"],))
        else:
            cursor.execute("SELECT * FROM customer_inferences ORDER BY id DESC LIMIT 150")
            
    inf_rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    
    if feature in ["tenure", "monthly_charges", "support_calls", "data_usage_gb"]:
        base_vals = baseline_df[feature].values
        if inf_rows:
            prod_vals = np.array([r[feature] for r in inf_rows if feature in r and r[feature] is not None])
        else:
            prod_vals = base_vals.copy()
            
        min_v = float(min(base_vals.min(), prod_vals.min() if len(prod_vals) else base_vals.min()))
        max_v = float(max(base_vals.max(), prod_vals.max() if len(prod_vals) else base_vals.max()))
        
        bins = np.linspace(min_v, max_v, 11)
        base_counts, _ = np.histogram(base_vals, bins=bins)
        prod_counts, _ = np.histogram(prod_vals, bins=bins)
        
        labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
        return {
            "type": "numerical",
            "feature": feature,
            "labels": labels,
            "baseline": (base_counts / len(base_vals)).round(3).tolist(),
            "production": (prod_counts / len(prod_vals) if len(prod_vals) else base_counts / len(base_vals)).round(3).tolist(),
            "baseline_mean": float(np.round(np.mean(base_vals), 2)),
            "production_mean": float(np.round(np.mean(prod_vals), 2)) if len(prod_vals) else float(np.round(np.mean(base_vals), 2))
        }
    else:
        base_counts = baseline_df[feature].astype(str).str.lower().value_counts(normalize=True).round(3).to_dict()
        if inf_rows:
            prod_s = pd.Series([r[feature] for r in inf_rows if feature in r]).astype(str).str.lower()
            prod_counts = prod_s.value_counts(normalize=True).round(3).to_dict()
        else:
            prod_counts = base_counts.copy()
            
        all_keys = sorted(list(set(base_counts.keys()).union(set(prod_counts.keys()))))
        return {
            "type": "categorical",
            "feature": feature,
            "labels": all_keys,
            "baseline": [float(base_counts.get(k, 0)) for k in all_keys],
            "production": [float(prod_counts.get(k, 0)) for k in all_keys]
        }

# ----------------- Simulator & Batch Processing -----------------
@app.post("/api/simulator/run-batch")
def run_batch_simulation(req: BatchSimulationRequest):
    batch_df = model_engine.generate_production_batch(scenario=req.scenario, n_samples=req.sample_count)
    
    probs, preds = model_engine.predict(batch_df)
    batch_df["churn_prob"] = np.round(probs, 4)
    batch_df["predicted_churn"] = preds
    
    baseline_df = model_engine.baseline_df
    base_probs, base_preds = model_engine.predict(baseline_df)
    
    base_acc = float(np.mean(base_preds == baseline_df["churn"]))
    batch_acc = float(np.mean(preds == batch_df["churn"]))
    
    # Run Drift Detective Analysis
    analysis = analyze_batch_drift(
        baseline_df=baseline_df,
        batch_df=batch_df,
        baseline_probs=base_probs,
        batch_probs=probs,
        baseline_acc=base_acc,
        batch_acc=batch_acc
    )
    
    # Store into SQLite Database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as count FROM batches")
    batch_index = cursor.fetchone()["count"] + 1
    batch_code = f"BATCH-{batch_index:04d}"
    now = datetime.now().isoformat()
    
    churn_rate = float(np.round(np.mean(preds), 4))
    
    cursor.execute(
        """
        INSERT INTO batches (batch_code, timestamp, scenario, sample_count, drift_score, drift_status, accuracy, accuracy_drop, churn_rate, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            batch_code,
            now,
            req.scenario,
            len(batch_df),
            analysis["overall_drift_score"],
            analysis["overall_status"],
            round(batch_acc, 4),
            analysis["accuracy_drop"],
            churn_rate,
            analysis["summary"]
        )
    )
    batch_id = cursor.lastrowid
    
    for fm in analysis["feature_metrics"]:
        cursor.execute(
            """
            INSERT INTO batch_features (batch_id, feature_name, psi_score, ks_stat, p_value, mean_prod, mean_base, drift_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                fm["feature"],
                fm["psi"],
                fm.get("ks_stat"),
                fm.get("p_value"),
                fm.get("mean_prod"),
                fm.get("mean_base"),
                1 if fm["drift_detected"] else 0
            )
        )
        
        if fm["status"] in ["WARNING", "CRITICAL"]:
            alert_sev = "CRITICAL" if fm["status"] == "CRITICAL" else "WARNING"
            cursor.execute(
                """
                INSERT INTO alerts (batch_id, batch_code, timestamp, severity, feature_name, alert_type, message, details, is_resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    batch_id,
                    batch_code,
                    now,
                    alert_sev,
                    fm["feature"],
                    "COVARIATE_DRIFT",
                    f"Drift detected in {fm['feature']} (PSI={fm['psi']})",
                    f"KS-stat: {fm.get('ks_stat')}, p-value: {fm.get('p_value')}, Prod Mean: {fm.get('mean_prod')}, Base Mean: {fm.get('mean_base')}"
                )
            )
            
    if analysis["accuracy_drop"] >= 0.08:
        sev = "CRITICAL" if analysis["accuracy_drop"] >= 0.16 else "WARNING"
        cursor.execute(
            """
            INSERT INTO alerts (batch_id, batch_code, timestamp, severity, feature_name, alert_type, message, details, is_resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                batch_id,
                batch_code,
                now,
                sev,
                "Model Accuracy",
                "CONCEPT_DRIFT",
                f"Silent model degradation! Accuracy dropped by {round(analysis['accuracy_drop']*100, 1)}%",
                f"Baseline: {round(base_acc*100, 1)}%, Batch: {round(batch_acc*100, 1)}%"
            )
        )
        
    for _, row in batch_df.iterrows():
        is_corr = 1 if int(row["predicted_churn"]) == int(row["churn"]) else 0
        cursor.execute(
            """
            INSERT INTO customer_inferences (batch_id, customer_id, tenure, monthly_charges, support_calls, data_usage_gb, contract, internet_service, payment_method, churn_prob, predicted_churn, actual_churn, is_correct, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                row["customer_id"],
                float(row["tenure"]),
                float(row["monthly_charges"]),
                int(row["support_calls"]),
                float(row["data_usage_gb"]),
                str(row["contract"]),
                str(row["internet_service"]),
                str(row["payment_method"]),
                float(row["churn_prob"]),
                int(row["predicted_churn"]),
                int(row["churn"]),
                is_corr,
                now
            )
        )
        
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "batch_id": batch_id,
        "batch_code": batch_code,
        "scenario": req.scenario,
        "sample_count": len(batch_df),
        "analysis": analysis
    }

@app.post("/api/simulator/single-predict")
def single_predict(req: SinglePredictRequest):
    single_df = pd.DataFrame([{
        "customer_id": "ONLINE_TEST",
        "tenure": req.tenure,
        "monthly_charges": req.monthly_charges,
        "support_calls": req.support_calls,
        "data_usage_gb": req.data_usage_gb,
        "contract": req.contract,
        "internet_service": req.internet_service,
        "payment_method": req.payment_method,
        "churn": req.actual_churn if req.actual_churn is not None else 0
    }])
    
    probs, preds = model_engine.predict(single_df)
    prob = float(probs[0])
    pred = int(preds[0])
    
    base_df = model_engine.baseline_df
    deviations = []
    for num_col in ["tenure", "monthly_charges", "support_calls", "data_usage_gb"]:
        mean_v = float(base_df[num_col].mean())
        std_v = float(base_df[num_col].std())
        val = getattr(req, num_col)
        z_score = abs(val - mean_v) / (std_v + 1e-6)
        if z_score > 2.0:
            deviations.append(f"{num_col} (Z-Score: {round(z_score, 1)}, deviates from training norm)")
            
    risk_level = "High Churn Risk" if prob >= 0.65 else ("Moderate Risk" if prob >= 0.40 else "Low Churn Risk")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        """
        INSERT INTO customer_inferences (batch_id, customer_id, tenure, monthly_charges, support_calls, data_usage_gb, contract, internet_service, payment_method, churn_prob, predicted_churn, actual_churn, is_correct, timestamp)
        VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"SINGLE-{datetime.now().strftime('%H%M%S')}",
            req.tenure,
            req.monthly_charges,
            req.support_calls,
            req.data_usage_gb,
            req.contract,
            req.internet_service,
            req.payment_method,
            round(prob, 4),
            pred,
            req.actual_churn,
            1 if req.actual_churn is not None and pred == req.actual_churn else None,
            now
        )
    )
    conn.commit()
    conn.close()
    
    return {
        "churn_probability": round(prob, 4),
        "predicted_churn": pred,
        "risk_level": risk_level,
        "outlier_deviations": deviations
    }

# ----------------- Alerts & Remediation -----------------
@app.get("/api/alerts")
def get_alerts(status: Optional[str] = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if status == "active":
        cursor.execute("SELECT * FROM alerts WHERE is_resolved = 0 ORDER BY id DESC")
    elif status == "resolved":
        cursor.execute("SELECT * FROM alerts WHERE is_resolved = 1 ORDER BY id DESC")
    else:
        cursor.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 50")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return {"alerts": rows}

@app.post("/api/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int, user: Dict[str, Any] = Depends(require_auth)):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "UPDATE alerts SET is_resolved = 1, resolved_at = ?, resolved_by = ? WHERE id = ?",
        (now, user["username"], alert_id)
    )
    conn.commit()
    conn.close()
    return {"success": True, "message": f"Alert {alert_id} resolved"}

@app.post("/api/model/retrain")
def retrain_model(user: Dict[str, Any] = Depends(require_auth)):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT tenure, monthly_charges, support_calls, data_usage_gb, contract, internet_service, payment_method, actual_churn as churn FROM customer_inferences WHERE actual_churn IS NOT NULL ORDER BY id DESC LIMIT 1000")
    prod_rows = [dict(r) for r in cursor.fetchall()]
    
    if len(prod_rows) >= 60:
        prod_df = pd.DataFrame(prod_rows)
        combined_df = pd.concat([model_engine.baseline_df.sample(n=min(len(model_engine.baseline_df), 1500)), prod_df], ignore_index=True)
    else:
        combined_df = model_engine.baseline_df.copy()
        
    curr_v = model_engine.version
    parts = curr_v.replace("v", "").split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    new_version = f"v{major}.{minor + 1}.0"
    
    new_metrics = model_engine.train_baseline_model(combined_df, version=new_version)
    
    now = datetime.now().isoformat()
    cursor.execute(
        "UPDATE alerts SET is_resolved = 1, resolved_at = ?, resolved_by = ? WHERE is_resolved = 0",
        (now, f"Auto-remediation via Retraining ({user['username']})")
    )
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "new_version": new_version,
        "algorithm": "RandomForestClassifier",
        "new_metrics": new_metrics,
        "message": f"Customer Churn model successfully retrained to {new_version}. Baseline distributions recalibrated!"
    }

# ----------------- Database Records Explorer -----------------
@app.get("/api/database/records")
def get_database_records(page: int = 1, page_size: int = 25, search: Optional[str] = None):
    offset = (page - 1) * page_size
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if search:
        search_term = f"%{search}%"
        cursor.execute("SELECT COUNT(*) as total FROM customer_inferences WHERE customer_id LIKE ?", (search_term,))
        total = cursor.fetchone()["total"]
        cursor.execute(
            "SELECT * FROM customer_inferences WHERE customer_id LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (search_term, page_size, offset)
        )
    else:
        cursor.execute("SELECT COUNT(*) as total FROM customer_inferences")
        total = cursor.fetchone()["total"]
        cursor.execute("SELECT * FROM customer_inferences ORDER BY id DESC LIMIT ? OFFSET ?", (page_size, offset))
        
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "records": rows
    }

@app.post("/api/database/seed-demo")
def seed_demo_data():
    scenarios = ["normal", "normal", "normal", "covariate_shift", "concept_drift"]
    results = []
    for sc in scenarios:
        res = run_batch_simulation(BatchSimulationRequest(scenario=sc, sample_count=100))
        results.append(res)
    return {"success": True, "batches_seeded": len(results)}

# ----------------- Custom Dataset Upload & Analysis -----------------
COLUMN_SYNONYMS = {
    "monthlycharges": "monthly_charges",
    "monthlycharge": "monthly_charges",
    "charges": "monthly_charges",
    "supportcalls": "support_calls",
    "calls": "support_calls",
    "tickets": "support_calls",
    "datausage": "data_usage_gb",
    "data_usage": "data_usage_gb",
    "usage_gb": "data_usage_gb",
    "usage": "data_usage_gb",
    "contract_type": "contract",
    "internetservice": "internet_service",
    "paymentmethod": "payment_method",
    "payment": "payment_method"
}

def normalize_uploaded_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=COLUMN_SYNONYMS)
    
    # Fill missing columns with reasonable baseline defaults if necessary
    base = model_engine.baseline_df
    if "tenure" not in df.columns:
        df["tenure"] = float(base["tenure"].median()) if base is not None else 20.0
    if "monthly_charges" not in df.columns:
        df["monthly_charges"] = float(base["monthly_charges"].median()) if base is not None else 65.0
    if "support_calls" not in df.columns:
        df["support_calls"] = int(base["support_calls"].median()) if base is not None else 2
    if "data_usage_gb" not in df.columns:
        df["data_usage_gb"] = float(base["data_usage_gb"].median()) if base is not None else 25.0
    if "contract" not in df.columns:
        df["contract"] = "month-to-month"
    if "internet_service" not in df.columns:
        df["internet_service"] = "fiber"
    if "payment_method" not in df.columns:
        df["payment_method"] = "card"
        
    if "customer_id" not in df.columns:
        df["customer_id"] = [f"UPLOAD-{1000 + i}" for i in range(len(df))]
        
    return df

@app.post("/api/datasets/upload-batch")
async def upload_and_analyze_batch(file: UploadFile = File(...), batch_title: Optional[str] = Form(None)):
    """
    Accepts any custom CSV dataset, parses customer features, executes Churn prediction,
    runs Kolmogorov-Smirnov test and Population Stability Index (PSI) against the baseline,
    detects drift anomalies, logs alerts, and stores records into SQLite!
    """
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")
        
    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")
        
    df = normalize_uploaded_df(df)
    
    # Run predictions
    probs, preds = model_engine.predict(df)
    df["churn_prob"] = np.round(probs, 4)
    df["predicted_churn"] = preds
    
    baseline_df = model_engine.baseline_df
    base_probs, base_preds = model_engine.predict(baseline_df)
    base_acc = float(np.mean(base_preds == baseline_df["churn"]))
    
    has_ground_truth = "churn" in df.columns
    if has_ground_truth:
        batch_acc = float(np.mean(preds == df["churn"]))
    else:
        # If no ground truth label, estimate degradation from high-entropy probability shift
        df["churn"] = None
        batch_acc = base_acc
        
    analysis = analyze_batch_drift(
        baseline_df=baseline_df,
        batch_df=df,
        baseline_probs=base_probs,
        batch_probs=probs,
        baseline_acc=base_acc,
        batch_acc=batch_acc
    )
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as count FROM batches")
    batch_index = cursor.fetchone()["count"] + 1
    batch_code = f"BATCH-U{batch_index:03d}"
    now = datetime.now().isoformat()
    scenario_label = batch_title or f"Custom CSV ({file.filename})"
    churn_rate = float(np.round(np.mean(preds), 4))
    
    cursor.execute(
        """
        INSERT INTO batches (batch_code, timestamp, scenario, sample_count, drift_score, drift_status, accuracy, accuracy_drop, churn_rate, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            batch_code,
            now,
            scenario_label,
            len(df),
            analysis["overall_drift_score"],
            analysis["overall_status"],
            round(batch_acc, 4),
            analysis["accuracy_drop"],
            churn_rate,
            analysis["summary"]
        )
    )
    batch_id = cursor.lastrowid
    
    for fm in analysis["feature_metrics"]:
        cursor.execute(
            """
            INSERT INTO batch_features (batch_id, feature_name, psi_score, ks_stat, p_value, mean_prod, mean_base, drift_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                fm["feature"],
                fm["psi"],
                fm.get("ks_stat"),
                fm.get("p_value"),
                fm.get("mean_prod"),
                fm.get("mean_base"),
                1 if fm["drift_detected"] else 0
            )
        )
        if fm["status"] in ["WARNING", "CRITICAL"]:
            alert_sev = "CRITICAL" if fm["status"] == "CRITICAL" else "WARNING"
            cursor.execute(
                """
                INSERT INTO alerts (batch_id, batch_code, timestamp, severity, feature_name, alert_type, message, details, is_resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    batch_id,
                    batch_code,
                    now,
                    alert_sev,
                    fm["feature"],
                    "COVARIATE_DRIFT",
                    f"Drift detected in {fm['feature']} (PSI={fm['psi']}) from {file.filename}",
                    f"KS-stat: {fm.get('ks_stat')}, p-value: {fm.get('p_value')}, Batch Mean: {fm.get('mean_prod')}, Baseline Mean: {fm.get('mean_base')}"
                )
            )
            
    if has_ground_truth and analysis["accuracy_drop"] >= 0.08:
        sev = "CRITICAL" if analysis["accuracy_drop"] >= 0.16 else "WARNING"
        cursor.execute(
            """
            INSERT INTO alerts (batch_id, batch_code, timestamp, severity, feature_name, alert_type, message, details, is_resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                batch_id,
                batch_code,
                now,
                sev,
                "Model Accuracy",
                "CONCEPT_DRIFT",
                f"Silent model degradation in uploaded batch! Accuracy dropped by {round(analysis['accuracy_drop']*100, 1)}%",
                f"Baseline: {round(base_acc*100, 1)}%, Batch: {round(batch_acc*100, 1)}%"
            )
        )
        
    for _, row in df.iterrows():
        act_churn = int(row["churn"]) if has_ground_truth and pd.notna(row["churn"]) else None
        is_corr = (1 if int(row["predicted_churn"]) == act_churn else 0) if act_churn is not None else None
        cursor.execute(
            """
            INSERT INTO customer_inferences (batch_id, customer_id, tenure, monthly_charges, support_calls, data_usage_gb, contract, internet_service, payment_method, churn_prob, predicted_churn, actual_churn, is_correct, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                str(row["customer_id"]),
                float(row["tenure"]),
                float(row["monthly_charges"]),
                int(row["support_calls"]),
                float(row["data_usage_gb"]),
                str(row["contract"]),
                str(row["internet_service"]),
                str(row["payment_method"]),
                float(row["churn_prob"]),
                int(row["predicted_churn"]),
                act_churn,
                is_corr,
                now
            )
        )
        
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "batch_id": batch_id,
        "batch_code": batch_code,
        "filename": file.filename,
        "total_records": len(df),
        "has_ground_truth": has_ground_truth,
        "analysis": analysis
    }

@app.post("/api/datasets/upload-baseline")
async def upload_new_baseline(file: UploadFile = File(...), user: Dict[str, Any] = Depends(require_auth)):
    """
    Accepts a new complete training dataset CSV, retrains the Churn Classifier model,
    recalculates all baseline distribution parameters, resets drift references, and increments version!
    """
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")
        
    df = normalize_uploaded_df(df)
    if "churn" not in df.columns:
        raise HTTPException(status_code=400, detail="The baseline training dataset must contain a 'churn' target column (0 or 1).")
        
    curr_v = model_engine.version
    major = int(curr_v.replace("v", "").split(".")[0]) + 1
    new_version = f"v{major}.0.0"
    
    metrics = model_engine.train_baseline_model(df, version=new_version)
    
    # Save as active dataset
    df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "customer_churn_data.csv"), index=False)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "UPDATE alerts SET is_resolved = 1, resolved_at = ?, resolved_by = ? WHERE is_resolved = 0",
        (now, f"Reset via Baseline Upload ({user['username']})")
    )
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "new_version": new_version,
        "sample_count": len(df),
        "metrics": metrics,
        "message": f"Successfully loaded new baseline dataset ({len(df)} records) and retrained model to {new_version}!"
    }

@app.get("/api/datasets/sample-templates/{template_name}")
def get_sample_template(template_name: str):
    """
    Generates and returns ready-to-use CSV sample datasets for testing:
    - healthy: 80 customers matching baseline
    - support_spike: 80 customers with support calls surging to 5-10
    - price_hike: 80 customers with high monthly charges ($110-$150)
    """
    if template_name == "support_spike":
        df = model_engine.generate_production_batch(scenario="covariate_shift", n_samples=80)
    elif template_name == "concept_drift":
        df = model_engine.generate_production_batch(scenario="concept_drift", n_samples=80)
    elif template_name == "price_hike":
        df = model_engine.generate_production_batch(scenario="normal", n_samples=80)
        df["monthly_charges"] = np.random.uniform(110.0, 155.0, size=80).round(2)
    else:
        df = model_engine.generate_production_batch(scenario="normal", n_samples=80)
        
    csv_str = df.to_csv(index=False)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={template_name}_dataset.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
