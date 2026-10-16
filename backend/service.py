"""Framework-independent core: loads artifacts and analyses a batch for drift."""
import json, os, threading
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import drift as D
from train import ART, train


class DriftService:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_batch = None
        self.history = []              # one entry per analysed batch -> timeline chart
        self.load()

    def load(self):
        self.model = joblib.load(os.path.join(ART, "model.joblib"))
        self.cfg = json.load(open(os.path.join(ART, "config.json")))
        self.num, self.cat = self.cfg["num_cols"], self.cfg["cat_cols"]
        self.target = self.cfg["target"]
        self.ref_full = pd.read_csv(os.path.join(ART, "reference.csv"))   # features + label
        self.ref = self.ref_full[self.num + self.cat]
        self.ref_proba = self.model.predict_proba(self.ref)[:, 1]

    def analyze(self, batch: pd.DataFrame, label="batch"):
        feats = self.num + self.cat
        missing = [c for c in feats if c not in batch.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self.last_batch = batch.copy()
        cur = batch[feats]
        proba = self.model.predict_proba(cur)[:, 1]

        fd = D.feature_drift(self.ref, cur, self.num, self.cat)
        pred_psi = D.psi_numeric(self.ref_proba, proba)

        perf = None
        drop = 0.0
        if self.target in batch.columns:       # labels arrive late in real life -> optional
            y = batch[self.target]
            perf = dict(accuracy=round(accuracy_score(y, proba > .5), 4),
                        f1=round(f1_score(y, proba > .5), 4),
                        auc=round(roc_auc_score(y, proba), 4))
            drop = self.cfg["baseline"]["auc"] - perf["auc"]      # AUC: robust to class-balance shifts

        level, score = D.overall_status(fd, pred_psi, drop)
        dists = {f["feature"]: D.distribution(self.ref[f["feature"]].values if f["type"] == "numeric"
                                              else self.ref[f["feature"]].astype(str).values,
                                              cur[f["feature"]].values if f["type"] == "numeric"
                                              else cur[f["feature"]].astype(str).values,
                                              f["type"] == "numeric") for f in fd}
        dists["prediction_score"] = D.distribution(self.ref_proba, proba, True)

        result = dict(label=label, level=level, score=score, n_rows=len(batch),
                      features=sorted(fd, key=lambda f: -f["psi"]),
                      prediction=dict(psi=round(pred_psi, 4), status=D.status_from_psi(pred_psi),
                                      ref_mean=round(float(self.ref_proba.mean()), 4),
                                      cur_mean=round(float(proba.mean()), 4)),
                      baseline=self.cfg["baseline"], performance=perf,
                      auc_drop=round(drop, 4) if perf else None,
                      distributions=dists,
                      recommendations=self._recommend(level, fd, pred_psi, perf, drop))
        self.history.append(dict(label=label, score=score, level=level,
                                 accuracy=perf["accuracy"] if perf else None))
        result["history"] = self.history[-20:]
        return result

    @staticmethod
    def _recommend(level, fd, pred_psi, perf, drop):
        top = [f["feature"] for f in fd if f["status"] == "drift"][:3]
        r = []
        if level == "healthy":
            return ["No action needed. Keep monitoring."]
        if top:
            r.append(f"Data drift detected in: {', '.join(top)}. Check upstream pipelines / business changes.")
        if pred_psi >= D.PSI_WARN:
            r.append("Model output distribution has shifted - review decision thresholds.")
        if perf is None:
            r.append("No ground-truth labels in this batch - collect labels to confirm performance impact.")
        elif drop >= 0.03:
            r.append(f"AUC fell by {drop*100:.1f} pts vs baseline - concept drift likely.")
        if level == "critical":
            r.append("Recommended: retrain the model on recent labelled data.")
        return r

    def retrain(self):
        """Close the loop: retrain on old reference + the latest labelled production batch."""
        if self.last_batch is None or self.target not in self.last_batch.columns:
            raise ValueError("Analyse a labelled batch first.")
        cols = self.num + self.cat + [self.target]
        full = pd.concat([self.ref_full[cols], self.last_batch[cols]], ignore_index=True)
        train(full, self.num, self.cat, self.target, save=True)
        self.load()
        self.history.append(dict(label="RETRAINED", score=0, level="healthy",
                                 accuracy=self.cfg["baseline"]["accuracy"]))
        return self.cfg["baseline"]
