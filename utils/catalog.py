"""
catalog.py
================================================================================
Product-level data for the recommendation tabs: the demo catalogue (articles_sample.csv, Kaggle
PART 1) with its photo flag, per-product sales statistics and sample type.

  units, revenue, units_last_90d  real sales of the product over the whole dataset
  units_rank_pct                  percentile of the product's units among ALL products of its own topic
  sample_type                     stratified (random 5% per topic) | customer_purchase | exemplar
                                  -> shares of product types / colours / prices are computed on the
                                  stratified sample only, because the other two are chosen, not random
================================================================================
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import load_articles, image_path
from utils import kaggle_data as kd
from utils.exemplars import top_exemplars

SALES_COLS = ["units", "revenue", "units_last_90d", "units_rank_pct"]


@st.cache_data(show_spinner=False)
def products() -> pd.DataFrame:
    df = load_articles().copy()
    df["topic"] = df["dominant_intention"].astype(int)
    df["has_image"] = [os.path.exists(image_path(a)) for a in df["article_id"]]
    for c in SALES_COLS:
        if c not in df.columns:
            df[c] = np.nan
    if "sample_type" not in df.columns:
        df["sample_type"] = "unknown"
    df["prod_name"] = df["prod_name"].fillna("").astype(str) if "prod_name" in df.columns else df["article_id"]
    return df


def has_sales(df: pd.DataFrame) -> bool:
    return bool(df["units"].notna().any())


def price_base() -> float:
    """Median transaction price in the dataset's normalised units = 'price index 100'."""
    v = kd.overview().get("price", {}).get("median")
    return float(v) if v else float("nan")


def price_index(price) -> float:
    b = price_base()
    try:
        return float(price) / b * 100 if b == b and b > 0 else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def top_products(df: pd.DataFrame, k: int, n: int, by: str = "typical") -> pd.DataFrame:
    """n products of topic k that have a photo. by: typical | bestsellers | slowest."""
    seg = df[(df["topic"] == k) & df["has_image"]]
    if by == "typical" or not has_sales(seg):
        return top_exemplars(seg, k, n)
    if by == "bestsellers":
        seg = seg.sort_values(["units", "revenue"], ascending=False, kind="mergesort")
    else:                                     # slowest: only stratified sample, so "slow" is not a result of how products were chosen
        st_seg = seg[seg["sample_type"] == "stratified"]
        seg = (st_seg if len(st_seg) >= n else seg).sort_values(["units_rank_pct", "units"], ascending=True, kind="mergesort")
    return seg.drop_duplicates("prod_name").head(n)


def caption(row, with_sales: bool = False) -> str:
    name = str(row["prod_name"])[:26]
    pi = price_index(row.get("avg_price"))
    cap = f"{name} · price {pi:.0f}" if pi == pi else name
    if with_sales and row.get("units") == row.get("units"):
        cap += f" · {int(row['units']):,} sold"
    return cap
