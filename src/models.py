"""Models and the validation scheme that decides between them.

Two things here are deliberate and worth defending in review.

**The baseline is a rule, not a dummy classifier.** Before any model earns its
place, it has to beat what an analyst would do in an afternoon with a
spreadsheet: rank customers by how recently they bought. If gradient boosting
cannot beat "sort by recency", it has bought complexity with nothing.

**Cross-validation expands forward in time.** `KFold` would train on October
to predict June, which production cannot do; the resulting score would be
optimistic and useless. Each fold here trains on cutoffs 1..n and validates on
cutoff n+1 — the same shape as the real deployment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import config, metrics

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
class RecencyBaseline(BaseEstimator, ClassifierMixin):
    """Rank by recency, the way an analyst would without a model.

    Scores are `1 / (1 + recency_days)`, which is monotonically decreasing in
    recency — the exact ordering a spreadsheet sort produces. It fits nothing,
    which is the point: it is the bar every model has to clear.
    """

    def __init__(self, column: str = "recency_days"):
        self.column = column

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        self.column_index_ = list(X.columns).index(self.column) if hasattr(X, "columns") else 0
        return self

    def predict_proba(self, X):
        recency = X[self.column].to_numpy() if hasattr(X, "columns") else X[:, self.column_index_]
        score = 1.0 / (1.0 + np.nan_to_num(recency, nan=999.0))
        return np.column_stack([1 - score, score])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
def _numeric_pipeline(features: list) -> ColumnTransformer:
    """Impute then scale. Both are fitted inside CV, never on the full data."""
    return ColumnTransformer(
        [("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), features)],
        remainder="drop",
    )


def build_models(features: list) -> dict:
    """The candidates, from 'a rule' to 'gradient boosting'.

    class_weight / scale_pos_weight are left at their defaults on purpose.
    Re-weighting shifts the probability scale, and this task is scored on
    ranking and calibration rather than on a 0.5 threshold, so re-weighting
    would trade away calibration for a metric nobody here optimises.
    """
    return {
        "recency_baseline": RecencyBaseline(),

        "logistic_regression": Pipeline([
            ("prep", _numeric_pipeline(features)),
            ("model", LogisticRegression(
                max_iter=2000, C=1.0, random_state=config.RANDOM_STATE)),
        ]),

        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400, max_depth=12, min_samples_leaf=20,
                n_jobs=-1, random_state=config.RANDOM_STATE)),
        ]),

        # Hyperparameters chosen by `tune_xgboost()` below, which searches
        # inside the same expanding-window CV. Tuning against the holdout
        # would turn the holdout into a training set.
        "xgboost": XGBClassifier(**XGB_PARAMS, eval_metric="logloss",
                                 random_state=config.RANDOM_STATE, n_jobs=-1),
    }


# Best configuration found by the search in `tune_xgboost`. Shallow trees with
# a heavy minimum child weight: the signal here is smooth, and depth mostly
# bought variance.
XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    min_child_weight=10, reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8,
)

XGB_GRID = [
    dict(n_estimators=300, max_depth=3, learning_rate=0.05, min_child_weight=10,
         reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=600, max_depth=3, learning_rate=0.03, min_child_weight=20,
         reg_lambda=5.0, subsample=0.8, colsample_bytree=0.6),
    dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
         reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=800, max_depth=2, learning_rate=0.05, min_child_weight=20,
         reg_lambda=10.0, subsample=0.9, colsample_bytree=0.6),
    dict(n_estimators=300, max_depth=6, learning_rate=0.05, min_child_weight=30,
         reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8),
]


def tune_xgboost(snapshots: pd.DataFrame, features: list,
                 cutoffs: list | None = None) -> pd.DataFrame:
    """Grid search scored by mean PR-AUC across the expanding-window folds."""
    from sklearn.metrics import average_precision_score

    cutoffs = cutoffs or config.DEV_CUTOFFS
    rows = []
    for i, params in enumerate(XGB_GRID):
        scores = []
        for train_cuts, valid_cut in expanding_window_folds(cutoffs, 3):
            train = snapshots[snapshots["cutoff"].astype(str).str[:10].isin(train_cuts)]
            valid = snapshots[snapshots["cutoff"].astype(str).str[:10] == valid_cut]
            model = XGBClassifier(**params, eval_metric="logloss",
                                  random_state=config.RANDOM_STATE, n_jobs=-1)
            model.fit(train[features], train["label"])
            scores.append(average_precision_score(
                valid["label"], model.predict_proba(valid[features])[:, 1]))
        rows.append({"config": i, **params,
                     "pr_auc_mean": float(np.mean(scores)),
                     "pr_auc_std": float(np.std(scores))})
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)


# --------------------------------------------------------------------------- #
# Temporal cross-validation
# --------------------------------------------------------------------------- #
@dataclass
class FoldResult:
    model: str
    fold: int
    train_cutoffs: list
    valid_cutoff: str
    n_train: int
    n_valid: int
    scores: dict


def expanding_window_folds(cutoffs: list, min_train: int = 3):
    """Yield (train_cutoffs, valid_cutoff) growing forward through time."""
    for i in range(min_train, len(cutoffs)):
        yield cutoffs[:i], cutoffs[i]


def cross_validate(
    snapshots: pd.DataFrame,
    features: list,
    cutoffs: list | None = None,
    min_train: int = 3,
) -> pd.DataFrame:
    """Run every model through every expanding-window fold."""
    cutoffs = cutoffs or config.DEV_CUTOFFS
    results = []

    for fold, (train_cuts, valid_cut) in enumerate(
        expanding_window_folds(cutoffs, min_train), start=1
    ):
        train = snapshots[snapshots["cutoff"].astype(str).str[:10].isin(train_cuts)]
        valid = snapshots[snapshots["cutoff"].astype(str).str[:10] == valid_cut]
        if train.empty or valid.empty:
            continue

        X_train, y_train = train[features], train["label"]
        X_valid, y_valid = valid[features], valid["label"]

        for name, model in build_models(features).items():
            model.fit(X_train, y_train)
            scores = metrics.evaluate(y_valid, model.predict_proba(X_valid)[:, 1])
            results.append(FoldResult(
                model=name, fold=fold, train_cutoffs=train_cuts, valid_cutoff=valid_cut,
                n_train=len(train), n_valid=len(valid), scores=scores,
            ))
            logger.info("fold %d %-20s valid=%s  pr_auc=%.3f  lift@20=%.2f",
                        fold, name, valid_cut, scores["pr_auc"],
                        scores[f"lift_at_{int(config.CAMPAIGN_CAPACITY*100)}"])

    rows = [{"model": r.model, "fold": r.fold, "valid_cutoff": r.valid_cutoff,
             "n_train": r.n_train, "n_valid": r.n_valid, **r.scores} for r in results]
    return pd.DataFrame(rows)


def summarise_cv(cv: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread per model. Spread matters: a model that wins on average
    but swings wildly across folds is a model that will surprise someone."""
    key = f"lift_at_{int(config.CAMPAIGN_CAPACITY*100)}"
    summary = cv.groupby("model").agg(
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        lift_mean=(key, "mean"), lift_std=(key, "std"),
        precision_at_20=(f"precision_at_{int(config.CAMPAIGN_CAPACITY*100)}", "mean"),
        brier_mean=("brier", "mean"),
        accuracy_mean=("accuracy", "mean"),
        majority_accuracy=("majority_accuracy", "mean"),
    ).sort_values("pr_auc_mean", ascending=False)
    return summary.round(4)
