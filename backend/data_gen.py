"""Synthetic customer-churn data (similar in spirit to IBM's Telco Churn sample).
`drift` in [0, 1] controls how far "production" data moves away from training data:
  - covariate drift : feature distributions shift
  - concept drift   : the relationship between features and churn changes
"""
import numpy as np
import pandas as pd

NUM_COLS = ["tenure", "monthly_charges", "support_calls", "data_usage_gb"]
CAT_COLS = ["contract", "internet_service", "payment_method"]
TARGET = "churn"


def _probs(base, delta, drift):
    p = np.clip(np.array(base) + drift * np.array(delta), 0.01, None)
    return p / p.sum()


def make_data(n=3000, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    tenure = np.clip(rng.gamma(2.0, 15, n) * (1 - 0.5 * drift), 1, 72)           # newer customers
    monthly = np.clip(rng.normal(65 + 25 * drift, 20 + 10 * drift, n), 15, 160)  # price hike
    calls = rng.poisson(1.5 + 2.5 * drift, n)                                    # more complaints
    usage = np.clip(rng.lognormal(3.0 + 0.4 * drift, 0.6, n), 1, 500)

    contract = rng.choice(["month-to-month", "one-year", "two-year"], n,
                          p=_probs([.55, .25, .20], [.25, -.10, -.15], drift))
    internet = rng.choice(["fiber", "dsl", "none"], n,
                          p=_probs([.45, .35, .20], [.20, -.10, -.10], drift))
    payment = rng.choice(["card", "bank", "cheque", "wallet"], n,
                         p=_probs([.35, .30, .20, .15], [-.10, -.10, -.05, .25], drift))

    # Concept drift: price matters LESS and support calls matter MORE as drift grows
    price_w = 0.03 * (1 - 2.5 * drift)      # at drift~0.8 the price effect has FLIPPED sign
    logit = (-1.2 + price_w * (monthly - 65) + (0.35 + 0.5 * drift) * calls
             - 0.03 * tenure + 1.0 * (contract == "month-to-month")
             + 0.5 * (internet == "fiber") + 0.004 * usage)
    logit = logit - logit.mean() - 0.8        # keep churn rate ~constant so only the *relationships* change
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame({"tenure": tenure.round(1), "monthly_charges": monthly.round(2),
                         "support_calls": calls, "data_usage_gb": usage.round(1),
                         "contract": contract, "internet_service": internet,
                         "payment_method": payment, TARGET: churn})
