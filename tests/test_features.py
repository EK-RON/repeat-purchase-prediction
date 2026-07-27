"""Leakage tests — the claim this whole project rests on.

A prediction project's headline number is only worth as much as its guarantee
that no feature saw the future. That guarantee is usually a comment. Here it
is an executable test: corrupt the outcome window, and assert that not one
feature value moves.

The fixtures are hand-written transactions with known cadences, so every
expected value below can be worked out on paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features as F

CUTOFF = pd.Timestamp("2011-06-01")


def make_transactions(rows) -> pd.DataFrame:
    """rows: (customer_id, date, invoice_no, stock_code, quantity, unit_price)"""
    frame = pd.DataFrame(rows, columns=[
        "customer_id", "invoice_date", "invoice_no", "stock_code", "quantity", "unit_price"])
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    frame["line_revenue"] = (frame["quantity"] * frame["unit_price"]).round(2)
    frame["country"] = "United Kingdom"
    return frame


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Four customers with deliberately different behaviour.

    1 — clockwork: orders every 30 days, and returns after the cutoff
    2 — lapsing: was regular, then went quiet months before the cutoff
    3 — one-off: a single order, never returns
    4 — new: first order just before the cutoff (should be filtered out)
    """
    rows = []
    for i, day in enumerate(["2011-01-01", "2011-02-01", "2011-03-03", "2011-04-02", "2011-05-02"]):
        rows.append((1, day, f"A{i}", "P1", 10, 2.0))
    rows.append((1, "2011-06-15", "A9", "P1", 10, 2.0))          # inside outcome window

    for i, day in enumerate(["2011-01-05", "2011-02-05", "2011-03-05"]):
        rows.append((2, day, f"B{i}", "P2", 5, 3.0))

    rows.append((3, "2011-02-10", "C0", "P3", 1, 50.0))
    rows.append((4, "2011-05-28", "D0", "P4", 2, 9.0))           # only 4 days of history
    return make_transactions(rows)


# --------------------------------------------------------------------------- #
# The leakage guarantee
# --------------------------------------------------------------------------- #
def test_features_are_identical_when_the_future_is_deleted(transactions):
    """The definitive test: remove the outcome window entirely.

    Features must not move by a single value. Labels must all become zero.
    If a feature had peeked forward, this test would catch it.
    """
    with_future = F.snapshot(transactions, CUTOFF, min_history_days=0)
    past_only = transactions[transactions["invoice_date"] < CUTOFF]
    without_future = F.snapshot(past_only, CUTOFF, min_history_days=0)

    columns = F.feature_columns(with_future)
    pd.testing.assert_frame_equal(
        with_future.set_index("customer_id")[columns].sort_index(),
        without_future.set_index("customer_id")[columns].sort_index(),
    )
    assert without_future["label"].sum() == 0
    assert with_future["label"].sum() == 1     # customer 1 returned


def test_features_are_identical_when_the_future_is_corrupted(transactions):
    """Inflate every post-cutoff value; features must be unmoved."""
    corrupted = transactions.copy()
    future = corrupted["invoice_date"] >= CUTOFF
    corrupted.loc[future, ["quantity", "unit_price", "line_revenue"]] *= 1000

    a = F.snapshot(transactions, CUTOFF, min_history_days=0)
    b = F.snapshot(corrupted, CUTOFF, min_history_days=0)
    columns = F.feature_columns(a)
    pd.testing.assert_frame_equal(
        a.set_index("customer_id")[columns].sort_index(),
        b.set_index("customer_id")[columns].sort_index(),
    )


def test_label_reflects_only_the_outcome_window(transactions):
    """An order after the horizon must not count as a return."""
    late = transactions.copy()
    late.loc[late["invoice_no"] == "A9", "invoice_date"] = pd.Timestamp("2011-08-01")

    snap = F.snapshot(late, CUTOFF, horizon_days=30, min_history_days=0)
    assert snap.set_index("customer_id").loc[1, "label"] == 0

    wide = F.snapshot(late, CUTOFF, horizon_days=90, min_history_days=0)
    assert wide.set_index("customer_id").loc[1, "label"] == 1


def test_cutoff_is_not_a_feature():
    """Calendar features would be extrapolation: the holdout months are unseen."""
    frame = pd.DataFrame(columns=["customer_id", "recency_days", "label", "cutoff"])
    assert F.feature_columns(frame) == ["recency_days"]


# --------------------------------------------------------------------------- #
# Feature correctness, on values that can be checked by hand
# --------------------------------------------------------------------------- #
def test_recency_and_frequency(transactions):
    snap = F.snapshot(transactions, CUTOFF, min_history_days=0).set_index("customer_id")
    # Customer 1's last pre-cutoff order is 2011-05-02, 30 days before 2011-06-01.
    assert snap.loc[1, "recency_days"] == 30
    assert snap.loc[1, "frequency"] == 5
    # Customer 2 went quiet on 2011-03-05.
    assert snap.loc[2, "recency_days"] == 88
    assert snap.loc[2, "frequency"] == 3


def test_recency_vs_gap_flags_the_lapsing_customer(transactions):
    """The feature that carries the most signal, on a case with a known answer."""
    snap = F.snapshot(transactions, CUTOFF, min_history_days=0).set_index("customer_id")
    # Customer 1 buys every ~30 days and is 30 days quiet -> right on schedule.
    assert snap.loc[1, "recency_vs_gap"] == pytest.approx(1.0, abs=0.15)
    # Customer 2 bought every ~31 days and has been quiet 88 -> nearly 3x overdue.
    assert snap.loc[2, "recency_vs_gap"] > 2.5


def test_single_order_customer_has_no_cadence(transactions):
    """One order means no gap exists; the feature must be NaN, not a fake zero."""
    snap = F.snapshot(transactions, CUTOFF, min_history_days=0).set_index("customer_id")
    assert np.isnan(snap.loc[3, "mean_gap_days"])
    assert snap.loc[3, "frequency"] == 1


def test_window_features_count_only_their_window(transactions):
    snap = F.snapshot(transactions, CUTOFF, min_history_days=0).set_index("customer_id")
    # Customer 1: one order in the 30 days before the cutoff (2011-05-02).
    assert snap.loc[1, "orders_last_30d"] == 1
    # Three in the 90 days (Mar 3, Apr 2, May 2).
    assert snap.loc[1, "orders_last_90d"] == 3
    # Customer 2 has been quiet for 88 days: nothing in the last 30.
    assert snap.loc[2, "orders_last_30d"] == 0


def test_min_history_filters_brand_new_customers(transactions):
    """Customer 4 has four days of history — not enough for cadence features."""
    snap = F.snapshot(transactions, CUTOFF, min_history_days=30)
    assert 4 not in set(snap["customer_id"])
    assert 3 in set(snap["customer_id"])


def test_snapshot_grain_is_one_row_per_customer(transactions):
    snap = F.snapshot(transactions, CUTOFF, min_history_days=0)
    assert snap["customer_id"].is_unique


def test_build_snapshots_stacks_cutoffs(transactions, tmp_path, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "SNAPSHOTS", tmp_path / "s.parquet")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)

    stacked = F.build_snapshots(transactions, cutoffs=["2011-04-01", "2011-05-01"])
    assert set(stacked["cutoff"].astype(str).str[:10]) == {"2011-04-01", "2011-05-01"}
    # Grain is (customer, cutoff), so a customer may appear once per cutoff.
    assert not stacked.duplicated(["customer_id", "cutoff"]).any()
