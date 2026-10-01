"""Probability-quality metrics. Betting needs calibrated probabilities, not just picks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def score(y, p) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    return {"n": len(y), "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "accuracy": accuracy_score(y, p > 0.5), "auc": roc_auc_score(y, p), "ece": ece(y, p)}


def paired_logloss_diff(y, p_a, p_b) -> tuple[float, float]:
    """mean(logloss_a - logloss_b) and its standard error (negative => a is better)."""
    def ll(p):
        p = np.clip(p, EPS, 1 - EPS)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d = ll(p_a) - ll(p_b)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
