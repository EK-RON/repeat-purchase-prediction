"""Final evaluation on the holdout — the part you only get to do once.

The development cutoffs (March–September) chose the model and its
hyperparameters. October and November were never looked at during that
process, so the numbers here are the closest thing this dataset offers to an
honest estimate of deployed performance.

Three things are measured beyond the headline score:

* **Per-cutoff results**, because a single averaged number hides the fact
  that base rates move with the season.
* **Calibration**, because turning a score into a budget decision requires a
  probability that means what it says.
* **Permutation importance**, computed on the holdout, because a tree's
  built-in `gain` importance is biased towards high-cardinality features and
  says nothing about whether the feature helps on unseen data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from src import config, metrics

logger = logging.getLogger(__name__)


def split_dev_test(snapshots: pd.DataFrame) -> tuple:
    """Development and holdout frames, split strictly by cutoff date."""
    keys = snapshots["cutoff"].astype(str).str[:10]
    return snapshots[keys.isin(config.DEV_CUTOFFS)], snapshots[keys.isin(config.TEST_CUTOFFS)]


def fit_on_dev(model, dev: pd.DataFrame, features: list):
    model.fit(dev[features], dev["label"])
    return model


def evaluate_holdout(models: dict, dev: pd.DataFrame, test: pd.DataFrame,
                     features: list) -> pd.DataFrame:
    """Fit each model once on all development snapshots, score the holdout."""
    rows = []
    for name, model in models.items():
        fit_on_dev(model, dev, features)
        scores = model.predict_proba(test[features])[:, 1]
        rows.append({"model": name, "split": "holdout (Oct+Nov)",
                     **metrics.evaluate(test["label"], scores)})

        # Per-cutoff, because seasonality moves the base rate underneath us.
        for cutoff, group in test.groupby(test["cutoff"].astype(str).str[:10]):
            group_scores = model.predict_proba(group[features])[:, 1]
            rows.append({"model": name, "split": cutoff,
                         **metrics.evaluate(group["label"], group_scores)})
    return pd.DataFrame(rows)


def calibration_table(y_true, y_score, bins: int = 10) -> pd.DataFrame:
    """Predicted probability versus observed frequency, bucket by bucket."""
    observed, predicted = calibration_curve(y_true, y_score, n_bins=bins, strategy="quantile")
    return pd.DataFrame({
        "predicted": predicted.round(3),
        "observed": observed.round(3),
        "gap": (predicted - observed).round(3),
    })


def permutation_table(model, test: pd.DataFrame, features: list,
                      n_repeats: int = 10) -> pd.DataFrame:
    """Drop in holdout PR-AUC when each feature is shuffled.

    Scored with average precision rather than accuracy, so importance is
    measured against the metric the project actually cares about.
    """
    result = permutation_importance(
        model, test[features], test["label"],
        scoring="average_precision", n_repeats=n_repeats,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "feature": features,
            "pr_auc_drop": result.importances_mean.round(5),
            "std": result.importances_std.round(5),
        })
        .sort_values("pr_auc_drop", ascending=False)
        .reset_index(drop=True)
    )


def capacity_sweep(y_true, y_score, capacities=(0.05, 0.10, 0.20, 0.30, 0.50)) -> pd.DataFrame:
    """Precision, recall and net value at each possible campaign size.

    This is the table that answers "how many people should we contact",
    which is the actual decision the model exists to support.
    """
    rows = []
    for k in capacities:
        row = metrics.expected_value(y_true, y_score, k)
        row["recall"] = metrics.recall_at_k(y_true, y_score, k)
        row["lift"] = metrics.lift_at_k(y_true, y_score, k)
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def seasonality_check(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Base rate by cutoff — the reason a single averaged score misleads."""
    keys = snapshots["cutoff"].astype(str).str[:10]
    out = snapshots.groupby(keys).agg(
        customers=("label", "size"), returned=("label", "sum"), base_rate=("label", "mean"),
    )
    out["split"] = np.where(out.index.isin(config.TEST_CUTOFFS), "holdout", "development")
    out["base_rate"] = (100 * out["base_rate"]).round(1)
    return out.reset_index(names="cutoff")


def stability_check(model, dev: pd.DataFrame, test: pd.DataFrame,
                    features: list) -> pd.DataFrame:
    """Does the model degrade as it predicts further from its training data?

    Fitting once and scoring two consecutive months answers a question every
    deployment eventually asks: how often does this need retraining?
    """
    fit_on_dev(model, dev, features)
    rows = []
    for cutoff, group in test.groupby(test["cutoff"].astype(str).str[:10]):
        scores = model.predict_proba(group[features])[:, 1]
        rows.append({
            "cutoff": cutoff,
            "months_after_training": len(rows) + 1,
            "base_rate": round(float(group["label"].mean()), 3),
            "pr_auc": round(average_precision_score(group["label"], scores), 4),
            "lift_at_20": round(metrics.lift_at_k(group["label"], scores), 3),
            "mean_predicted": round(float(scores.mean()), 3),
        })
    return pd.DataFrame(rows)
