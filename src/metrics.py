"""Metrics — chosen to match the decision, not to flatter the model.

The campaign can contact a fixed share of the base each month. That single
sentence rules out most of the standard metric menu:

* **Accuracy** answers "how often is the label right at threshold 0.5", a
  question nobody asked. With a 29% base rate, predicting "nobody returns"
  scores 71%. It is reported here precisely so that number is visible.
* **ROC-AUC** measures ranking across the whole list, including the bottom
  half the campaign will never touch. It is useful for comparing models but
  it is not the objective.
* **PR-AUC (average precision)** is the honest summary for an imbalanced
  ranking task, because it is computed against the positive class rather
  than being propped up by the many easy negatives.
* **Precision@k and lift@k** are the metric of the actual decision: of the
  20% of customers we contact, what share do we get right, and how much
  better is that than contacting 20% at random.
* **Brier score and calibration** matter because the money question is
  expected value per contact, and expected value needs a probability that
  means what it says — not merely a correct ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from src import config


def precision_at_k(y_true, y_score, k: float = config.CAMPAIGN_CAPACITY) -> float:
    """Precision among the top k share of scored customers."""
    y_true = np.asarray(y_true)
    n = max(1, int(round(len(y_true) * k)))
    top = np.argsort(np.asarray(y_score))[::-1][:n]
    return float(y_true[top].mean())


def recall_at_k(y_true, y_score, k: float = config.CAMPAIGN_CAPACITY) -> float:
    """Share of all returning customers captured inside the top k."""
    y_true = np.asarray(y_true)
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    n = max(1, int(round(len(y_true) * k)))
    top = np.argsort(np.asarray(y_score))[::-1][:n]
    return float(y_true[top].sum() / positives)


def lift_at_k(y_true, y_score, k: float = config.CAMPAIGN_CAPACITY) -> float:
    """How many times better than random targeting the same number of people."""
    base = float(np.asarray(y_true).mean())
    return precision_at_k(y_true, y_score, k) / base if base else float("nan")


def evaluate(y_true, y_score, threshold: float = 0.5,
             k: float = config.CAMPAIGN_CAPACITY) -> dict:
    """Every metric in one dict, so comparisons are always like-for-like."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    return {
        "base_rate": float(y_true.mean()),
        # Reported to be argued with, not to be optimised.
        "accuracy": accuracy_score(y_true, y_pred),
        "majority_accuracy": max(float(y_true.mean()), 1 - float(y_true.mean())),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        f"precision_at_{int(k*100)}": precision_at_k(y_true, y_score, k),
        f"recall_at_{int(k*100)}": recall_at_k(y_true, y_score, k),
        f"lift_at_{int(k*100)}": lift_at_k(y_true, y_score, k),
        "brier": brier_score_loss(y_true, y_score),
    }


def gains_table(y_true, y_score, bins: int = 10) -> pd.DataFrame:
    """Decile table: the artefact a marketing team actually reads."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
    frame["decile"] = (np.arange(len(frame)) * bins // len(frame)) + 1

    base = frame["y"].mean()
    out = frame.groupby("decile").agg(
        customers=("y", "size"), returned=("y", "sum"), precision=("y", "mean"),
    )
    out["lift"] = out["precision"] / base
    out["cumulative_returned"] = out["returned"].cumsum()
    out["cumulative_recall"] = out["cumulative_returned"] / frame["y"].sum()
    out["cumulative_precision"] = out["cumulative_returned"] / out["customers"].cumsum()
    return out.round(3)


def expected_value(
    y_true, y_score, k: float,
    contact_cost: float = config.CONTACT_COST,
    margin: float = config.INCREMENTAL_MARGIN,
) -> dict:
    """Turn a ranking into money, under explicit and challengeable assumptions.

    The assumption doing the most work is that contacting a customer who was
    going to return anyway still earns the margin. That is generous; a real
    programme would need an incrementality test (holdout group) to measure
    the uplift rather than the outcome. It is stated here rather than buried.
    """
    y_true = np.asarray(y_true)
    n = max(1, int(round(len(y_true) * k)))
    top = np.argsort(np.asarray(y_score))[::-1][:n]
    hits = int(y_true[top].sum())

    revenue = hits * margin
    cost = n * contact_cost
    return {
        "k": k, "contacted": n, "hits": hits,
        "precision": hits / n, "revenue": revenue, "cost": cost,
        "net": revenue - cost,
        "net_per_contact": (revenue - cost) / n,
    }
