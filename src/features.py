"""Feature engineering under strict temporal discipline.

This is where prediction projects usually go wrong, so the rule is enforced
structurally rather than remembered:

    Every feature is computed from `transactions[invoice_date < cutoff]`.
    Every label is computed from `transactions[cutoff <= invoice_date < cutoff + H]`.

Those two frames are built by separate functions that never see each other's
input. A feature *cannot* accidentally look forward, because the future rows
are not in the DataFrame it is given. A test asserts this by corrupting the
future and checking that no feature moves.

The output is a "snapshot" table: one row per (customer, cutoff) pair, which
is the shape every churn/propensity model actually trains on.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Feature block builders — each takes ONLY the past
# --------------------------------------------------------------------------- #
def _order_level(past: pd.DataFrame) -> pd.DataFrame:
    """Collapse lines to orders once; several feature blocks need this."""
    return (
        past.groupby(["customer_id", "invoice_no"], observed=True)
        .agg(order_date=("invoice_date", "min"),
             order_value=("line_revenue", "sum"),
             order_items=("quantity", "sum"),
             order_products=("stock_code", "nunique"))
        .reset_index()
    )


def _rfm_block(orders: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Recency, frequency, monetary and tenure — the classical core."""
    grouped = orders.groupby("customer_id", observed=True)
    out = grouped.agg(
        last_order=("order_date", "max"),
        first_order=("order_date", "min"),
        frequency=("invoice_no", "nunique"),
        monetary=("order_value", "sum"),
        avg_order_value=("order_value", "mean"),
        std_order_value=("order_value", "std"),
        max_order_value=("order_value", "max"),
        avg_items=("order_items", "mean"),
        avg_products=("order_products", "mean"),
    )
    out["recency_days"] = (cutoff - out["last_order"]).dt.days
    out["tenure_days"] = (cutoff - out["first_order"]).dt.days
    out["std_order_value"] = out["std_order_value"].fillna(0.0)
    # Revenue per day of relationship: separates a big-but-old customer from a
    # steadily active one, which raw monetary cannot.
    out["revenue_per_tenure_day"] = out["monetary"] / out["tenure_days"].clip(lower=1)
    return out.drop(columns=["last_order", "first_order"])


def _cadence_block(orders: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """How regularly does this customer buy, and are they overdue?

    `recency_vs_gap` is the single most useful feature in this project: a
    customer 40 days quiet means nothing on its own, but 40 days quiet when
    they normally return every 20 is a customer slipping away.
    """
    ordered = orders.sort_values(["customer_id", "order_date"])
    gaps = ordered.groupby("customer_id", observed=True)["order_date"].diff().dt.days

    frame = pd.DataFrame({"customer_id": ordered["customer_id"], "gap": gaps})
    stats = frame.groupby("customer_id", observed=True)["gap"].agg(
        mean_gap_days="mean", std_gap_days="std", min_gap_days="min", max_gap_days="max",
    )
    stats["std_gap_days"] = stats["std_gap_days"].fillna(0.0)
    # Coefficient of variation: is the cadence clockwork or erratic?
    stats["gap_cv"] = (stats["std_gap_days"] / stats["mean_gap_days"]).replace(
        [np.inf, -np.inf], np.nan
    )

    last = orders.groupby("customer_id", observed=True)["order_date"].max()
    recency = (cutoff - last).dt.days
    stats["recency_vs_gap"] = recency / stats["mean_gap_days"].clip(lower=1)
    return stats


def _window_block(orders: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Activity in trailing windows, plus momentum between them."""
    out = pd.DataFrame(index=orders["customer_id"].unique())
    out.index.name = "customer_id"

    for days in (30, 60, 90, 180):
        window = orders[orders["order_date"] >= cutoff - pd.Timedelta(days=days)]
        agg = window.groupby("customer_id", observed=True).agg(
            **{f"orders_last_{days}d": ("invoice_no", "nunique"),
               f"revenue_last_{days}d": ("order_value", "sum")}
        )
        out = out.join(agg)

    out = out.fillna(0.0)
    # Momentum: is recent activity a rising or falling share of the quarter?
    out["order_momentum"] = out["orders_last_30d"] / out["orders_last_90d"].clip(lower=1)
    out["revenue_momentum"] = out["revenue_last_30d"] / out["revenue_last_90d"].clip(lower=1)
    # Is the customer accelerating relative to their own half-year?
    out["orders_90d_vs_180d"] = out["orders_last_90d"] / out["orders_last_180d"].clip(lower=1)
    return out


def _breadth_block(past: pd.DataFrame) -> pd.DataFrame:
    """Catalogue breadth: a customer buying across many products is stickier."""
    grouped = past.groupby("customer_id", observed=True)
    out = grouped.agg(
        distinct_products=("stock_code", "nunique"),
        total_units=("quantity", "sum"),
        avg_unit_price=("unit_price", "mean"),
    )
    orders = grouped["invoice_no"].nunique()
    out["products_per_order"] = out["distinct_products"] / orders.clip(lower=1)
    return out


def _market_block(past: pd.DataFrame) -> pd.DataFrame:
    """Where the customer trades. Kept minimal because the base is 90% UK."""
    country = past.groupby("customer_id", observed=True)["country"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else None
    )
    return pd.DataFrame({
        "is_uk": (country == "United Kingdom").astype(int),
        "is_export": (country != "United Kingdom").astype(int),
    })


FEATURE_BLOCKS = ("rfm", "cadence", "window", "breadth", "market")


def build_features(past: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Assemble every feature block for one cutoff, from past transactions only.

    `past` must already be filtered to `invoice_date < cutoff`; `snapshot()`
    is what guarantees that, and it is the only caller in the pipeline.
    """
    orders = _order_level(past)
    features = (
        _rfm_block(orders, cutoff)
        .join(_cadence_block(orders, cutoff))
        .join(_window_block(orders, cutoff))
        .join(_breadth_block(past))
        .join(_market_block(past))
    )
    # A first-time buyer has no gap to speak of; NaN here means "unknown
    # cadence", which the models handle explicitly rather than by imputation.
    return features


# --------------------------------------------------------------------------- #
# Labels — computed from the future, and only the future
# --------------------------------------------------------------------------- #
def build_labels(future: pd.DataFrame, customers: pd.Index) -> pd.Series:
    """1 if the customer placed at least one order in the outcome window."""
    buyers = set(future["customer_id"].unique())
    return pd.Series([int(c in buyers) for c in customers], index=customers, name="label")


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #
def snapshot(
    transactions: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    horizon_days: int = config.HORIZON_DAYS,
    min_history_days: int = config.MIN_HISTORY_DAYS,
) -> pd.DataFrame:
    """One (customer, cutoff) table: features from the past, label from the future.

    The two slices below are the entire leakage argument of this project. They
    are disjoint by construction, and nothing downstream can re-join them.
    """
    cutoff = pd.Timestamp(cutoff)
    outcome_end = cutoff + pd.Timedelta(days=horizon_days)

    past = transactions[transactions["invoice_date"] < cutoff]
    future = transactions[
        (transactions["invoice_date"] >= cutoff) & (transactions["invoice_date"] < outcome_end)
    ]
    if past.empty:
        raise ValueError(f"No history before {cutoff:%Y-%m-%d}")

    features = build_features(past, cutoff)

    # Eligibility: enough history for the features to mean anything.
    eligible = features[features["tenure_days"] >= min_history_days]
    labels = build_labels(future, eligible.index)

    out = eligible.copy()
    out["label"] = labels
    out["cutoff"] = cutoff
    return out.reset_index()


def build_snapshots(
    transactions: pd.DataFrame,
    cutoffs: list | None = None,
    horizon_days: int = config.HORIZON_DAYS,
) -> pd.DataFrame:
    """Stack every cutoff into one training table."""
    cutoffs = cutoffs or config.CUTOFFS
    frames = []
    for cutoff in cutoffs:
        frame = snapshot(transactions, cutoff, horizon_days=horizon_days)
        logger.info("%s: %5d customers, base rate %.1f%%",
                    pd.Timestamp(cutoff).date(), len(frame), 100 * frame["label"].mean())
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.SNAPSHOTS, index=False)
    return out


def feature_columns(snapshots: pd.DataFrame) -> list:
    """Everything except identifiers, the label and the cutoff.

    `cutoff` is deliberately NOT a feature. The holdout months (October,
    November) never appear in training, so a calendar feature could only be
    extrapolated — and a model that has learned "November is busy" from
    training data it never saw is a model that has learned nothing.
    """
    excluded = {"customer_id", "label", "cutoff"}
    return [c for c in snapshots.columns if c not in excluded]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from src.data import build_transactions

    tx = build_transactions()
    snaps = build_snapshots(tx)
    print(f"\n{len(snaps):,} snapshot rows | {len(feature_columns(snaps))} features")
    print(f"overall base rate: {100 * snaps['label'].mean():.1f}%")
