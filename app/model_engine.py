import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from app.database import get_db_connection, init_db

# Ensure database tables exist
init_db()

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_FILE = os.path.join(MODEL_DIR, "churn_model.pkl")
BASELINE_FILE = os.path.join(MODEL_DIR, "baseline_data.pkl")
CSV_PATH = os.path.join(MODEL_DIR, "customer_churn_data.csv")

NUMERICAL_FEATURES = ["tenure", "monthly_charges", "support_calls", "data_usage_gb"]
CATEGORICAL_FEATURES = ["contract", "internet_service", "payment_method"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

CATEGORICAL_ENCODINGS = {
    "contract": {"month-to-month": 0, "one-year": 1, "two-year": 2},
    "internet_service": {"dsl": 0, "fiber": 1, "none": 2},
    "payment_method": {"bank": 0, "card": 1, "cheque": 2, "wallet": 3}
}

def load_or_create_dataset() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        # Fallback if csv missing
        df = pd.DataFrame()
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = df.copy()
    for col, mapping in CATEGORICAL_ENCODINGS.items():
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].astype(str).str.lower().map(mapping).fillna(0).astype(int)
    return df_enc[ALL_FEATURES]

class ModelEngine:
    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.baseline_df: Optional[pd.DataFrame] = None
        self.version: str = "v1.0.0"
        self.metrics: Dict[str, float] = {}
        self.load_or_train()

    def train_baseline_model(self, df: pd.DataFrame, version: str = "v1.0.0") -> Dict[str, float]:
        X = encode_features(df)
        y = df["churn"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob))
        }
        
        self.model = clf
        self.baseline_df = df.copy()
        self.version = version
        self.metrics = metrics
        
        # Save artifacts
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump({"model": clf, "version": version, "metrics": metrics}, f)
        with open(BASELINE_FILE, "wb") as f:
            pickle.dump(df, f)
            
        # Log to DB
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE models SET is_active = 0")
        cursor.execute(
            """
            INSERT INTO models (version, algorithm, accuracy, precision, recall, f1, roc_auc, trained_at, training_samples, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                version,
                "RandomForestClassifier (100 trees, depth 8)",
                round(metrics["accuracy"], 4),
                round(metrics["precision"], 4),
                round(metrics["recall"], 4),
                round(metrics["f1"], 4),
                round(metrics["roc_auc"], 4),
                datetime.now().isoformat(),
                len(df)
            )
        )
        conn.commit()
        conn.close()
        
        return metrics

    def load_or_train(self):
        if os.path.exists(MODEL_FILE) and os.path.exists(BASELINE_FILE):
            try:
                with open(MODEL_FILE, "rb") as f:
                    saved = pickle.load(f)
                    self.model = saved["model"]
                    self.version = saved.get("version", "v1.0.0")
                    self.metrics = saved.get("metrics", {})
                with open(BASELINE_FILE, "rb") as f:
                    self.baseline_df = pickle.load(f)
                return
            except Exception as e:
                print(f"Error loading saved model: {e}. Retraining...")
        
        # Load from customer churn dataset CSV
        df = load_or_create_dataset()
        self.train_baseline_model(df, version="v1.0.0")

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = encode_features(df)
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return probs, preds

    def generate_production_batch(self, scenario: str = "normal", n_samples: int = 120, seed: Optional[int] = None) -> pd.DataFrame:
        if seed is None:
            seed = np.random.randint(1000, 999999)
        np.random.seed(seed)
        
        # Sample base records from baseline dataset
        sample_indices = np.random.choice(len(self.baseline_df), size=n_samples, replace=True)
        batch_df = self.baseline_df.iloc[sample_indices].copy().reset_index(drop=True)
        
        if scenario == "normal":
            # Subtle natural noise
            batch_df["tenure"] = np.clip(batch_df["tenure"] + np.random.normal(0, 1.5, n_samples), 1.0, 72.0).round(1)
            batch_df["monthly_charges"] = np.clip(batch_df["monthly_charges"] + np.random.normal(0, 2.0, n_samples), 15.0, 160.0).round(2)
            batch_df["data_usage_gb"] = np.clip(batch_df["data_usage_gb"] + np.random.normal(0, 2.0, n_samples), 2.0, 180.0).round(1)
            
        elif scenario == "covariate_shift":
            # Significant feature drift:
            # Support calls surge (customer support issues), data usage surges, contract shifts toward month-to-month
            batch_df["support_calls"] = np.random.poisson(lam=5.5, size=n_samples).clip(0, 15)
            batch_df["data_usage_gb"] = np.clip(batch_df["data_usage_gb"] * 1.6 + np.random.normal(15, 8, n_samples), 5.0, 195.0).round(1)
            batch_df["monthly_charges"] = np.clip(batch_df["monthly_charges"] * 1.25, 25.0, 160.0).round(2)
            batch_df["contract"] = np.random.choice(["month-to-month", "one-year", "two-year"], size=n_samples, p=[0.85, 0.12, 0.03])
            
        elif scenario == "concept_drift":
            # Relationship between features and churn flips / breaks!
            # Competitor aggressive marketing: long-tenure customers with 0-1 support calls and 2-year contracts
            # suddenly churn at a massive 80% rate! The model predicts they are safe, but they churn!
            new_churn = []
            for _, row in batch_df.iterrows():
                if row["contract"] in ["two-year", "one-year"] or row["tenure"] > 25.0:
                    new_churn.append(1 if np.random.rand() < 0.82 else 0)
                else:
                    new_churn.append(int(row["churn"]))
            batch_df["churn"] = new_churn
            
        elif scenario == "severe_shock":
            # Both massive covariate drift (support call explosion, high bills) and severe concept drift (outage)
            batch_df["support_calls"] = np.random.poisson(lam=7.0, size=n_samples).clip(0, 15)
            batch_df["monthly_charges"] = np.clip(batch_df["monthly_charges"] * 1.35, 30.0, 160.0).round(2)
            batch_df["contract"] = np.random.choice(["month-to-month", "one-year", "two-year"], size=n_samples, p=[0.90, 0.08, 0.02])
            batch_df["churn"] = (np.random.rand(n_samples) < 0.78).astype(int)
            
        batch_df["customer_id"] = [f"CUST-{np.random.randint(10000, 99999)}" for _ in range(n_samples)]
        return batch_df

model_engine = ModelEngine()
