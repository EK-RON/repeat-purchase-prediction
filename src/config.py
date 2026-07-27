"""Configuration — and, more importantly, the written problem definition.

Every modelling project needs one paragraph that a non-technical colleague can
check. Here it is:

    For each customer who has bought at least once before a given cutoff date,
    predict whether they will place at least one order in the 30 days
    *after* that date.

The 30-day horizon is a business choice, not a statistical one: the retention
campaign runs monthly, so a prediction that reaches beyond the next campaign
cycle cannot be acted on. Everything else in this file follows from that
sentence.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_XLSX = RAW_DIR / "online_retail.xlsx"
RAW_PARQUET = RAW_DIR / "online_retail_raw.parquet"
TRANSACTIONS = PROCESSED_DIR / "transactions.parquet"
SNAPSHOTS = PROCESSED_DIR / "snapshots.parquet"

# --------------------------------------------------------------------------- #
# Source — the same pinned extract as the other projects in this series
# --------------------------------------------------------------------------- #
SOURCE_URL = (
    "https://raw.githubusercontent.com/eaintkyawthmu/"
    "UCI_Online_Retail_Dataset_Cleaned_Version/master/Online%20Retail.xlsx"
)
SOURCE_SHA256 = "43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d"

# --------------------------------------------------------------------------- #
# The temporal design — the single most important decision in this project
# --------------------------------------------------------------------------- #
# A snapshot is one (customer, cutoff_date) pair. Features are built from
# transactions strictly BEFORE the cutoff; the label is read from the window
# strictly AFTER it. No feature may ever see the outcome window — that is the
# leakage this whole structure exists to prevent.
HORIZON_DAYS = 30

# Cutoffs are monthly, matching the campaign cadence. The first one leaves
# three months of history so that features such as inter-purchase gaps are
# meaningful; the last one leaves a full outcome window inside the data.
CUTOFFS = [
    "2011-03-01", "2011-04-01", "2011-05-01", "2011-06-01", "2011-07-01",
    "2011-08-01", "2011-09-01", "2011-10-01", "2011-11-01",
]

# The split is by TIME, not at random. A random split would let the model
# learn from October to predict June, which is not a thing production can do.
DEV_CUTOFFS = CUTOFFS[:7]      # 2011-03 .. 2011-09  — model selection, CV
TEST_CUTOFFS = CUTOFFS[7:]     # 2011-10, 2011-11    — touched once, at the end

# A customer must have at least this many days of history at the cutoff,
# otherwise "recency" and "average gap" are noise rather than signal.
MIN_HISTORY_DAYS = 30

# --------------------------------------------------------------------------- #
# Business framing for the decision threshold
# --------------------------------------------------------------------------- #
# The campaign has a fixed budget: it can contact this share of the eligible
# base each month. That makes the task a RANKING problem, and precision@k the
# metric that matches the decision actually being made.
CAMPAIGN_CAPACITY = 0.20       # contact the top 20% of scored customers

# Illustrative unit economics, used to turn scores into an expected-value
# decision. These are plausible for a wholesale giftware business, and they
# are assumptions, not facts — the sensitivity of the answer to them is
# reported rather than hidden.
CONTACT_COST = 1.50            # £ per customer contacted
INCREMENTAL_MARGIN = 28.0      # £ expected margin from a retained order

RANDOM_STATE = 42

PALETTE = {
    "primary": "#1f4e79",
    "accent": "#e07b39",
    "muted": "#9aa5b1",
    "good": "#2e7d5b",
    "bad": "#b3392f",
}
