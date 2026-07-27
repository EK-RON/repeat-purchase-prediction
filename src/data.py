"""Acquire the source extract and reduce it to a clean transaction table.

This project re-derives its own transaction table from the pinned source
rather than importing the output of the upstream EDA repository. The cleaning
rules are the same, deliberately: a model project that cannot be run without
first running another repository is a model project nobody will run.
"""

from __future__ import annotations

import hashlib
import logging

import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)
CHUNK = 1 << 20

# Stock codes that are not products (postage, fees, vouchers, adjustments).
NON_PRODUCT_CODES = {
    "POST", "DOT", "C2", "M", "m", "B", "D", "S", "PADS",
    "BANK CHARGES", "AMAZONFEE", "CRUK",
}


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool = False):
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = config.RAW_XLSX
    if target.exists() and not force and _sha256(target) == config.SOURCE_SHA256:
        logger.info("Source extract present and verified")
        return target

    logger.info("Downloading source extract")
    response = requests.get(config.SOURCE_URL, stream=True, timeout=120)
    response.raise_for_status()
    with open(target, "wb") as fh:
        for chunk in response.iter_content(CHUNK):
            fh.write(chunk)

    actual = _sha256(target)
    if actual != config.SOURCE_SHA256:
        raise RuntimeError(f"Checksum mismatch: expected {config.SOURCE_SHA256}, got {actual}")
    return target


def load_raw(force: bool = False) -> pd.DataFrame:
    if config.RAW_PARQUET.exists() and not force:
        return pd.read_parquet(config.RAW_PARQUET)

    path = download(force=force)
    logger.info("Parsing workbook (~1 minute)")
    frame = pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})
    frame["Description"] = frame["Description"].astype("string")
    frame.to_parquet(config.RAW_PARQUET, index=False)
    return frame


def build_transactions(raw: pd.DataFrame | None = None, force: bool = False) -> pd.DataFrame:
    """Return identified, valid product sales — the modelling universe.

    Rows removed here and why:
      * exact duplicates on the business key — the same event written twice
      * cancellations, negative quantities, zero prices — not a purchase
      * non-product lines (postage, fees, vouchers) — not a purchase either
      * rows with no customer id — a customer we cannot track cannot be scored
    """
    if config.TRANSACTIONS.exists() and not force and raw is None:
        return pd.read_parquet(config.TRANSACTIONS)

    raw = raw if raw is not None else load_raw(force=force)
    frame = raw.rename(columns={
        "InvoiceNo": "invoice_no", "StockCode": "stock_code", "Description": "description",
        "Quantity": "quantity", "InvoiceDate": "invoice_date", "UnitPrice": "unit_price",
        "CustomerID": "customer_id", "Country": "country",
    }).copy()

    for col in ("invoice_no", "stock_code", "country"):
        frame[col] = frame[col].astype("string").str.strip()
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    frame["customer_id"] = frame["customer_id"].astype("Int64")

    frame = frame.drop_duplicates(
        subset=["invoice_no", "stock_code", "quantity", "invoice_date",
                "unit_price", "customer_id", "country"]
    )

    is_product = ~frame["stock_code"].isin(NON_PRODUCT_CODES) & \
                 ~frame["stock_code"].str.startswith(("gift_", "DCGS"), na=False)
    valid = (
        is_product
        & ~frame["invoice_no"].str.startswith("C", na=False)
        & (frame["quantity"] > 0)
        & (frame["unit_price"] > 0)
        & frame["customer_id"].notna()
    )

    out = frame.loc[valid, ["customer_id", "invoice_no", "invoice_date",
                            "stock_code", "quantity", "unit_price", "country"]].copy()
    out["line_revenue"] = (out["quantity"] * out["unit_price"]).round(2)
    out = out.sort_values(["customer_id", "invoice_date"]).reset_index(drop=True)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.TRANSACTIONS, index=False)
    logger.info("Transactions: %s rows, %s customers",
                f"{len(out):,}", f"{out.customer_id.nunique():,}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tx = build_transactions()
    print(tx.head())
    print(f"\n{len(tx):,} rows | {tx.customer_id.nunique():,} customers | "
          f"{tx.invoice_date.min():%Y-%m-%d} to {tx.invoice_date.max():%Y-%m-%d}")
