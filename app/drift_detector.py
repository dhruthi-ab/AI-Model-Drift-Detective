import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple

def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10, epsilon: float = 1e-4) -> float:
    """
    Computes Population Stability Index (PSI) between baseline and current distributions.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate drift
    PSI >= 0.2: Significant drift
    """
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]
    
    if len(baseline) == 0 or len(current) == 0:
        return 0.0
        
    quantiles = np.linspace(0, 100, num_buckets + 1)
    try:
        breakpoints = np.percentile(baseline, quantiles)
    except Exception:
        breakpoints = np.linspace(np.min(baseline), np.max(baseline), num_buckets + 1)
        
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0
        
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    base_counts, _ = np.histogram(baseline, bins=breakpoints)
    curr_counts, _ = np.histogram(current, bins=breakpoints)
    
    base_pct = base_counts / len(baseline)
    curr_pct = curr_counts / len(current)
    
    base_pct = np.clip(base_pct, epsilon, 1.0)
    curr_pct = np.clip(curr_pct, epsilon, 1.0)
    
    base_pct = base_pct / np.sum(base_pct)
    curr_pct = curr_pct / np.sum(curr_pct)
    
    psi_value = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(np.round(psi_value, 4))

def calculate_categorical_psi(baseline_series: pd.Series, current_series: pd.Series, epsilon: float = 1e-4) -> float:
    """
    Computes PSI for categorical values.
    """
    base_s = baseline_series.dropna().astype(str).str.lower()
    curr_s = current_series.dropna().astype(str).str.lower()
    
    categories = list(set(base_s.unique()).union(set(curr_s.unique())))
    if not categories:
        return 0.0
        
    base_counts = base_s.value_counts(normalize=True).to_dict()
    curr_counts = curr_s.value_counts(normalize=True).to_dict()
    
    base_probs = np.array([base_counts.get(cat, epsilon) for cat in categories])
    curr_probs = np.array([curr_counts.get(cat, epsilon) for cat in categories])
    
    base_probs = base_probs / np.sum(base_probs)
    curr_probs = curr_probs / np.sum(curr_probs)
    
    psi_value = np.sum((curr_probs - base_probs) * np.log(curr_probs / base_probs))
    return float(np.round(psi_value, 4))

def analyze_batch_drift(baseline_df: pd.DataFrame, batch_df: pd.DataFrame, baseline_probs: np.ndarray, batch_probs: np.ndarray, baseline_acc: float, batch_acc: float) -> Dict[str, Any]:
    """
    Comprehensive drift detective analysis:
    - Feature-level covariate drift for user features: tenure, monthly_charges, support_calls, data_usage_gb, contract, internet_service, payment_method
    - Prediction drift (Output probability PSI & KS test)
    - Concept drift & Silent Accuracy Degradation
    """
    feature_metrics = []
    drifted_features = []
    
    # 1. Numerical features
    num_features = ["tenure", "monthly_charges", "support_calls", "data_usage_gb"]
    for feat in num_features:
        base_vals = baseline_df[feat].values
        prod_vals = batch_df[feat].values
        
        psi = calculate_psi(base_vals, prod_vals)
        ks_res = stats.ks_2samp(base_vals, prod_vals)
        w_dist = float(stats.wasserstein_distance(base_vals, prod_vals))
        
        mean_base = float(np.mean(base_vals))
        mean_prod = float(np.mean(prod_vals))
        
        is_drifted = (psi >= 0.15) or (ks_res.pvalue < 0.01 and psi >= 0.10)
        if is_drifted:
            drifted_features.append(feat)
            
        feature_metrics.append({
            "feature": feat,
            "type": "numerical",
            "psi": round(psi, 4),
            "ks_stat": round(float(ks_res.statistic), 4),
            "p_value": float(ks_res.pvalue),
            "wasserstein_dist": round(w_dist, 2),
            "mean_base": round(mean_base, 2),
            "mean_prod": round(mean_prod, 2),
            "std_base": round(float(np.std(base_vals)), 2),
            "std_prod": round(float(np.std(prod_vals)), 2),
            "drift_detected": bool(is_drifted),
            "status": "CRITICAL" if psi >= 0.25 else ("WARNING" if psi >= 0.10 else "STABLE")
        })

    # 2. Categorical features
    cat_features = ["contract", "internet_service", "payment_method"]
    for feat in cat_features:
        psi = calculate_categorical_psi(baseline_df[feat], batch_df[feat])
        is_drifted = psi >= 0.15
        if is_drifted:
            drifted_features.append(feat)
            
        base_dist = baseline_df[feat].astype(str).str.lower().value_counts(normalize=True).round(3).to_dict()
        prod_dist = batch_df[feat].astype(str).str.lower().value_counts(normalize=True).round(3).to_dict()
        
        feature_metrics.append({
            "feature": feat,
            "type": "categorical",
            "psi": round(psi, 4),
            "ks_stat": None,
            "p_value": None,
            "wasserstein_dist": None,
            "mean_base": None,
            "mean_prod": None,
            "base_distribution": base_dist,
            "prod_distribution": prod_dist,
            "drift_detected": bool(is_drifted),
            "status": "CRITICAL" if psi >= 0.25 else ("WARNING" if psi >= 0.10 else "STABLE")
        })

    # 3. Prediction Output Drift
    pred_psi = calculate_psi(baseline_probs, batch_probs)
    pred_ks = stats.ks_2samp(baseline_probs, batch_probs)
    
    # 4. Performance Degradation / Concept Drift
    acc_drop = round(baseline_acc - batch_acc, 4)
    concept_drift_detected = acc_drop >= 0.08
    
    # 5. Overall Drift Score
    max_feature_psi = max([m["psi"] for m in feature_metrics])
    overall_drift_score = round(float(0.4 * max_feature_psi + 0.3 * pred_psi + 0.3 * max(0.0, acc_drop * 2)), 4)
    
    if overall_drift_score >= 0.25 or acc_drop >= 0.16:
        overall_status = "CRITICAL_DRIFT"
    elif overall_drift_score >= 0.10 or acc_drop >= 0.07:
        overall_status = "WARNING_DRIFT"
    else:
        overall_status = "HEALTHY"

    # Root Cause Attribution
    reasons = []
    if drifted_features:
        reasons.append(f"Significant feature drift detected in: {', '.join(drifted_features)} (Max Feature PSI: {max_feature_psi}).")
    if pred_psi >= 0.15:
        reasons.append(f"Output churn probability distribution shifted significantly (Output PSI: {pred_psi}).")
    if concept_drift_detected:
        reasons.append(f"Silent model degradation! Accuracy plummeted by {round(acc_drop * 100, 1)}% (Baseline: {round(baseline_acc * 100, 1)}% -> Live Batch: {round(batch_acc * 100, 1)}%).")
    
    if not reasons:
        summary_text = f"Customer churn model is performing stably. Production inferences match baseline training distribution within statistical tolerance (Drift Index: {overall_drift_score})."
    else:
        summary_text = " ".join(reasons)

    return {
        "overall_status": overall_status,
        "overall_drift_score": overall_drift_score,
        "accuracy_drop": acc_drop,
        "baseline_accuracy": round(baseline_acc, 4),
        "batch_accuracy": round(batch_acc, 4),
        "prediction_psi": round(pred_psi, 4),
        "prediction_ks_pvalue": float(pred_ks.pvalue),
        "drifted_features": drifted_features,
        "feature_metrics": feature_metrics,
        "summary": summary_text
    }
