"""Tests for the metrics and the validation scheme.

These pin down the two things most likely to be quietly wrong in a modelling
repository: whether the business metrics compute what they claim, and whether
the cross-validation ever lets the model see the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, metrics, models


# --------------------------------------------------------------------------- #
# Ranking metrics
# --------------------------------------------------------------------------- #
def test_precision_at_k_takes_the_top_scores():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    # Top 20% is two customers, both positive.
    assert metrics.precision_at_k(y, scores, 0.2) == 1.0
    # Top 50% is five: two positives.
    assert metrics.precision_at_k(y, scores, 0.5) == pytest.approx(0.4)


def test_lift_is_precision_over_the_base_rate():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])          # base rate 0.3
    scores = np.linspace(1, 0, 10)
    assert metrics.lift_at_k(y, scores, 0.2) == pytest.approx(1.0 / 0.3)


def test_perfect_ranking_beats_random_ranking():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 1000)
    assert metrics.lift_at_k(y, y.astype(float), 0.2) > metrics.lift_at_k(y, rng.random(1000), 0.2)


def test_recall_at_k_is_share_of_positives_captured():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    scores = np.linspace(1, 0, 10)
    assert metrics.recall_at_k(y, scores, 0.2) == pytest.approx(0.5)   # 2 of 4


def test_metrics_are_invariant_to_monotone_rescaling():
    """Ranking metrics must not care about the scale of the scores."""
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, 500)
    scores = rng.random(500)
    assert metrics.precision_at_k(y, scores, 0.2) == \
           metrics.precision_at_k(y, scores * 100 + 5, 0.2)


# --------------------------------------------------------------------------- #
# The argument against accuracy, as an executable assertion
# --------------------------------------------------------------------------- #
def test_a_useless_model_still_scores_high_accuracy():
    """Predicting 'nobody returns' beats 70% accuracy at this base rate."""
    y = np.concatenate([np.ones(30), np.zeros(70)])
    always_zero = np.zeros(100)
    scored = metrics.evaluate(y, always_zero)

    assert scored["accuracy"] == pytest.approx(0.70)
    assert scored["majority_accuracy"] == pytest.approx(0.70)
    # ...while being worthless at the job the model exists to do.
    assert scored["recall"] == 0.0
    assert scored["pr_auc"] < 0.35


def test_gains_table_is_monotone_in_cumulative_recall():
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.3, 1000)
    scores = y * 0.5 + rng.random(1000) * 0.5      # informative but noisy
    gains = metrics.gains_table(y, scores)

    assert gains["cumulative_recall"].is_monotonic_increasing
    assert gains["cumulative_recall"].iloc[-1] == pytest.approx(1.0)
    assert gains["lift"].iloc[0] > gains["lift"].iloc[-1]


def test_expected_value_accounts_for_contact_cost():
    y = np.ones(100)
    scores = np.linspace(1, 0, 100)
    result = metrics.expected_value(y, scores, k=0.1, contact_cost=2.0, margin=10.0)

    assert result["contacted"] == 10
    assert result["hits"] == 10
    assert result["net"] == pytest.approx(10 * 10.0 - 10 * 2.0)


# --------------------------------------------------------------------------- #
# Validation scheme
# --------------------------------------------------------------------------- #
def test_expanding_window_never_trains_on_the_future():
    cutoffs = ["2011-03-01", "2011-04-01", "2011-05-01", "2011-06-01", "2011-07-01"]
    folds = list(models.expanding_window_folds(cutoffs, min_train=3))

    assert len(folds) == 2
    for train_cuts, valid_cut in folds:
        assert all(c < valid_cut for c in train_cuts), "a training cutoff came after validation"


def test_expanding_window_grows():
    cutoffs = [f"2011-0{i}-01" for i in range(1, 8)]
    sizes = [len(train) for train, _ in models.expanding_window_folds(cutoffs, min_train=3)]
    assert sizes == sorted(sizes) and sizes[0] == 3


def test_recency_baseline_ranks_by_recency():
    """The bar every model has to clear: it must at least sort correctly."""
    X = pd.DataFrame({"recency_days": [1, 10, 100], "other": [0, 0, 0]})
    baseline = models.RecencyBaseline().fit(X)
    scores = baseline.predict_proba(X)[:, 1]

    assert scores[0] > scores[1] > scores[2]
    assert np.all((scores >= 0) & (scores <= 1))


def test_baseline_handles_missing_recency():
    X = pd.DataFrame({"recency_days": [np.nan, 5.0]})
    scores = models.RecencyBaseline().fit(X).predict_proba(X)[:, 1]
    assert np.isfinite(scores).all()


def test_model_zoo_contains_the_documented_candidates():
    zoo = models.build_models(["recency_days"])
    assert set(zoo) == {"recency_baseline", "logistic_regression", "random_forest", "xgboost"}


def test_every_model_produces_valid_probabilities():
    rng = np.random.default_rng(3)
    features = ["a", "b", "recency_days"]
    X = pd.DataFrame(rng.random((200, 3)), columns=features)
    y = (X["a"] + rng.random(200) * 0.4 > 0.8).astype(int)

    for name, model in models.build_models(features).items():
        model.fit(X, y)
        proba = model.predict_proba(X)[:, 1]
        assert proba.shape == (200,), name
        assert np.all((proba >= 0) & (proba <= 1)), name


def test_models_tolerate_missing_values():
    """Cadence features are NaN for one-order customers; nothing may crash."""
    rng = np.random.default_rng(4)
    features = ["recency_days", "mean_gap_days"]
    X = pd.DataFrame({
        "recency_days": rng.random(120) * 100,
        "mean_gap_days": np.where(rng.random(120) < 0.3, np.nan, rng.random(120) * 40),
    })
    y = (X["recency_days"] < 50).astype(int)

    for name, model in models.build_models(features).items():
        model.fit(X, y)
        assert np.isfinite(model.predict_proba(X)[:, 1]).all(), name


def test_cross_validate_returns_a_row_per_model_and_fold():
    rng = np.random.default_rng(5)
    cutoffs = config.DEV_CUTOFFS[:5]
    frames = []
    for cutoff in cutoffs:
        n = 200
        frames.append(pd.DataFrame({
            "customer_id": np.arange(n),
            "recency_days": rng.random(n) * 100,
            "frequency": rng.integers(1, 10, n),
            "label": rng.binomial(1, 0.3, n),
            "cutoff": pd.Timestamp(cutoff),
        }))
    snapshots = pd.concat(frames, ignore_index=True)

    cv = models.cross_validate(snapshots, ["recency_days", "frequency"], cutoffs, min_train=3)
    assert set(cv["model"]) == {"recency_baseline", "logistic_regression",
                                "random_forest", "xgboost"}
    assert cv["fold"].nunique() == 2
    assert cv["pr_auc"].between(0, 1).all()
