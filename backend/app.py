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
    }

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
    """Profile any readable CSV; run drift analysis when model features are present."""
    try:
        raw = await file.read()
        if len(raw) > 50 * 1024 * 1024:
            raise HTTPException(413, "CSV is too large. Maximum size is 50 MB.")

        df = pd.read_csv(io.BytesIO(raw))
        expected = set(svc.num + svc.cat)

        if expected.issubset(set(df.columns)):
            return svc.analyze(df, label=file.filename)

        result = profile_dataset(df, file.filename)
        return {
            "mode": "profile",
            "label": result["label"],
            "rows": result["n_rows"],
            "column_count": result["n_columns"],
            "duplicate_rows": result["duplicate_rows"],
            "columns": [
                {
                    "name": c["name"],
                    "dtype": c["type"],
                    "missing": c["missing_count"],
                    "unique": c["unique_count"],
                    "summary": (
                        f"mean={c['mean']}, median={c['median']}, "
                        f"min={c['min']}, max={c['max']}, "
                        f"outliers={c['outlier_count']}"
                        if "mean" in c else ""
                    ),
                    "top_values": c.get("top_values", []),
                }
                for c in result["columns"]
            ],
        }
    except HTTPException:
        raise
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as e:
        raise HTTPException(400, f"Could not analyse CSV: {e}")


@app.post("/api/retrain")
def retrain():
    try:
        return dict(new_baseline=svc.retrain())
    except ValueError as e:
        raise HTTPException(400, str(e))


# SQLite Auth Endpoints
import hashlib, sqlite3
from pydantic import BaseModel
from typing import Optional

DB_FILE = os.path.join(os.path.dirname(__file__), "..", "churnguard_users.db")

class AuthPayload(BaseModel):
    name: Optional[str] = "MLOps Engineer"
    email: str
    password: str
    role: Optional[str] = "Lead MLOps Engineer"

def get_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_login TEXT,
        login_count INTEGER DEFAULT 0
    )
    """)
    conn.commit()
    return conn

@app.post("/api/auth/login")
def auth_login(payload: AuthPayload):
    conn = get_db()
    cursor = conn.cursor()
    email = payload.email.strip().lower()
    password_hash = hashlib.sha256(payload.password.strip().encode()).hexdigest()
    cursor.execute("SELECT id, name, email, role, password_hash, login_count FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    if row and row[4] == password_hash:
        now = datetime.now().isoformat()
        new_count = (row[5] or 0) + 1
        cursor.execute("UPDATE users SET last_login = ?, login_count = ? WHERE id = ?", (now, new_count, row[0]))
        conn.commit()
        conn.close()
        return {"success": True, "user": {"id": row[0], "name": row[1], "email": row[2], "role": row[3]}}
    conn.close()
    raise HTTPException(401, "Invalid email or password")

@app.post("/api/auth/register")
def auth_register(payload: AuthPayload):
    conn = get_db()
    cursor = conn.cursor()
    email = payload.email.strip().lower()
    name = (payload.name or "MLOps Engineer").strip()
    role = (payload.role or "Lead MLOps Engineer").strip()
    password_hash = hashlib.sha256(payload.password.strip().encode()).hexdigest()
    now = datetime.now().isoformat()
    try:
        cursor.execute("""
            INSERT INTO users (name, email, password_hash, role, created_at, last_login, login_count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (name, email, password_hash, role, now, now))
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return {"success": True, "user": {"id": user_id, "name": name, "email": email, "role": role}}
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(409, "Email already exists")

@app.get("/api/auth/users")
def auth_users():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, role, created_at, last_login, login_count FROM users ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return {"success": True, "users": [{ "id": r[0], "name": r[1], "email": r[2], "role": r[3], "createdAt": r[4], "lastLogin": r[5], "loginCount": r[6]} for r in rows]}


# Serve the dashboard at "/"
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend"),
                           html=True), name="ui")
