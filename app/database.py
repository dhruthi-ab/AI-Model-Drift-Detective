import sqlite3
import hashlib
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "drift_detective.db")

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    if salt is None:
        salt = os.urandom(16).hex()
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    return pwd_hash, salt

def verify_password(password: str, pwd_hash: str, salt: str) -> bool:
    check_hash, _ = hash_password(password, salt)
    return check_hash == pwd_hash

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        role TEXT DEFAULT 'analyst',
        created_at TEXT NOT NULL
    )
    """)

    # Active Sessions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        username TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)

    # Models Registry Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT UNIQUE NOT NULL,
        algorithm TEXT NOT NULL,
        accuracy REAL NOT NULL,
        precision REAL NOT NULL,
        recall REAL NOT NULL,
        f1 REAL NOT NULL,
        roc_auc REAL NOT NULL,
        trained_at TEXT NOT NULL,
        training_samples INTEGER NOT NULL,
        is_active INTEGER DEFAULT 1
    )
    """)

    # Monitoring Batches Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_code TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        scenario TEXT NOT NULL,
        sample_count INTEGER NOT NULL,
        drift_score REAL NOT NULL,
        drift_status TEXT NOT NULL,
        accuracy REAL,
        accuracy_drop REAL,
        churn_rate REAL,
        summary TEXT
    )
    """)

    # Batch Feature Drift Metrics
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS batch_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER NOT NULL,
        feature_name TEXT NOT NULL,
        psi_score REAL NOT NULL,
        ks_stat REAL,
        p_value REAL,
        mean_prod REAL,
        mean_base REAL,
        drift_detected INTEGER NOT NULL,
        FOREIGN KEY (batch_id) REFERENCES batches (id)
    )
    """)

    # Drift Alerts Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER,
        batch_code TEXT,
        timestamp TEXT NOT NULL,
        severity TEXT NOT NULL,
        feature_name TEXT,
        alert_type TEXT NOT NULL,
        message TEXT NOT NULL,
        details TEXT,
        is_resolved INTEGER DEFAULT 0,
        resolved_at TEXT,
        resolved_by TEXT
    )
    """)

    # Customer Inferences Table matching user's exact dataset
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customer_inferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER,
        customer_id TEXT NOT NULL,
        tenure REAL NOT NULL,
        monthly_charges REAL NOT NULL,
        support_calls INTEGER NOT NULL,
        data_usage_gb REAL NOT NULL,
        contract TEXT NOT NULL,
        internet_service TEXT NOT NULL,
        payment_method TEXT NOT NULL,
        churn_prob REAL NOT NULL,
        predicted_churn INTEGER NOT NULL,
        actual_churn INTEGER,
        is_correct INTEGER,
        timestamp TEXT NOT NULL
    )
    """)

    # Check if default admin user exists
    cursor.execute("SELECT id FROM users WHERE username = ?", ("admin",))
    if not cursor.fetchone():
        now = datetime.now().isoformat()
        admin_hash, admin_salt = hash_password("admin123")
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("admin", "admin@drift-detective.ai", admin_hash, admin_salt, "admin", now)
        )
        analyst_hash, analyst_salt = hash_password("analyst123")
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("analyst", "analyst@drift-detective.ai", analyst_hash, analyst_salt, "analyst", now)
        )

    conn.commit()
    conn.close()

def create_session(user_id: int, username: str, role: str) -> str:
    token = str(uuid.uuid4())
    now = datetime.now()
    expires_at = (now + timedelta(days=7)).isoformat()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (token, user_id, username, role, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
        (token, user_id, username, role, now.isoformat(), expires_at)
    )
    conn.commit()
    conn.close()
    return token

def get_session(token: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE token = ?", (token,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    
    expires_at = datetime.fromisoformat(row["expires_at"])
    if datetime.now() > expires_at:
        delete_session(token)
        return None
        
    return dict(row)

def delete_session(token: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()
