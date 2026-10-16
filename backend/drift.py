"""Drift statistics: PSI, KS test, chi-square, plus histogram helpers for the UI."""
import numpy as np
from scipy import stats

EPS = 1e-4
PSI_WARN, PSI_DRIFT = 0.10, 0.25   # industry-standard PSI thresholds


def _psi_from_props(r, c):
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _edges(ref, bins=10):
    e = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(e) < 3:                       # near-constant feature
        e = np.array([ref.min() - 1, ref.mean(), ref.max() + 1])
    e = e.astype(float)
    e[0], e[-1] = -np.inf, np.inf
    return e


def psi_numeric(ref, cur, bins=10):
    e = _edges(ref, bins)
    r = np.histogram(ref, e)[0] / len(ref)
    c = np.histogram(cur, e)[0] / len(cur)
    return _psi_from_props(r, c)


def psi_categorical(ref, cur):
    cats = sorted(set(ref) | set(cur))
    r = np.array([(ref == k).mean() for k in cats])
    c = np.array([(cur == k).mean() for k in cats])
    return _psi_from_props(r, c)


def status_from_psi(v):
    return "stable" if v < PSI_WARN else "warning" if v < PSI_DRIFT else "drift"


def feature_drift(ref_df, cur_df, num_cols, cat_cols):
    out = []
    for col in num_cols:
        r, c = ref_df[col].dropna().values, cur_df[col].dropna().values
        p = psi_numeric(r, c)
        ks = stats.ks_2samp(r, c)
        out.append(dict(feature=col, type="numeric", psi=round(p, 4), test="KS",
                        stat=round(float(ks.statistic), 4), p_value=float(ks.pvalue),
                        status=status_from_psi(p),
                        ref_mean=round(float(r.mean()), 3), cur_mean=round(float(c.mean()), 3)))
    for col in cat_cols:
        r, c = ref_df[col].astype(str), cur_df[col].astype(str)
        p = psi_categorical(r.values, c.values)
        cats = sorted(set(r) | set(c))
        table = np.array([[(r == k).sum() for k in cats], [(c == k).sum() for k in cats]])
        chi = stats.chi2_contingency(table)
        out.append(dict(feature=col, type="categorical", psi=round(p, 4), test="Chi-square",
                        stat=round(float(chi[0]), 3), p_value=float(chi[1]),
                        status=status_from_psi(p)))
    return out


def distribution(ref_vals, cur_vals, numeric, bins=10):
    """Aligned reference/current percentages for plotting."""
    if numeric:
        e = _edges(ref_vals, bins)
        r = np.histogram(ref_vals, e)[0] / len(ref_vals)
        c = np.histogram(cur_vals, e)[0] / len(cur_vals)
        inner = e[1:-1]
        labels = ([f"<{inner[0]:.2f}"] + [f"{inner[i]:.2f}-{inner[i+1]:.2f}" for i in range(len(inner) - 1)]
                  + [f">{inner[-1]:.2f}"])
    else:
        labels = sorted(set(ref_vals) | set(cur_vals))
        r = np.array([(ref_vals == k).mean() for k in labels])
        c = np.array([(cur_vals == k).mean() for k in labels])
    return dict(labels=labels, ref=(r * 100).round(2).tolist(), cur=(c * 100).round(2).tolist())


def overall_status(feats, pred_psi, perf_drop):
    """Combine feature drift, prediction drift and performance drop into one verdict."""
    n_drift = sum(f["status"] == "drift" for f in feats)
    n_warn = sum(f["status"] == "warning" for f in feats)
    share = n_drift / max(len(feats), 1)
    score = round(40 * share + 25 * min(pred_psi / PSI_DRIFT, 1)
                  + 35 * min(max(perf_drop, 0) / 0.10, 1), 1)
    if share >= 0.3 or pred_psi >= PSI_DRIFT or perf_drop >= 0.06:
        level = "critical"
    elif n_drift or n_warn or pred_psi >= PSI_WARN or perf_drop >= 0.03:
        level = "warning"
    else:
        level = "healthy"
    return level, score
