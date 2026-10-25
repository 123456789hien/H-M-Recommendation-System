"""
topicfacts.py
================================================================================
One table with every key figure of each of the 10 preference topics, assembled from the Kaggle outputs
(never typed in). Used by the first and third tabs.
================================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import kaggle_data as kd
from utils.data_loader import load_monthly_trends
from utils.preference_labels import preference_name, product_line
from utils.priority import priority_table
from utils.thesis_data import segment_static, model_improvement
from utils.trends import with_period_columns, aggregate_period
from utils.population import load_pop_table


@st.cache_data(show_spinner=False)
def topic_facts() -> pd.DataFrame:
    seg = load_pop_table("pop_segments")
    tier = load_pop_table("pop_price_tier")
    pst = kd.price_stats()
    gap_by_k = {k: segment_static(k)["gap"] for k in range(10)}
    prio = priority_table(aggregate_period(with_period_columns(load_monthly_trends()), "All time"), "All time", None, gap_by_k)
    base = float(kd.overview().get("price", {}).get("median") or np.nan)
    rows = []
    for k in range(10):
        s = segment_static(k)
        r = {"topic": k, "name": preference_name(k), "line": product_line(k),
             "customers": s["users"], "customer_share": s["user_share"], "catalogue_share": s["cat_share"],
             "purchase_share": s["demand_share"], "gap": s["gap"], "concentration": s["confidence"],
             "products": s["article_count"], "model_gain_pct": model_improvement(k)}
        p = pst[pst["topic"] == k] if not pst.empty else pst
        r["price_tier"] = str(p["price_tier"].iloc[0]) if len(p) and "price_tier" in p else ""
        r["mean_product_price"] = float(p["mean"].iloc[0]) if len(p) and "mean" in p else np.nan
        if not seg.empty:
            q = seg[seg["intention"] == k]
            if len(q):
                q = q.iloc[0]
                r["revenue_share"] = float(q.get("revenue_share_pct", np.nan))
                r["avg_price"] = float(q.get("avg_price", np.nan))
                r["purchases"] = int(q.get("n_purchases", 0))
                r["buyers"] = int(q.get("n_buyers", 0))
                r["loyal_pct"] = float(q.get("customers_10plus_pct", np.nan))
                r["mean_age"] = float(q.get("mean_age", np.nan))
                r["purchases_per_buyer"] = r["purchases"] / r["buyers"] if r["buyers"] else np.nan
                r["spend_per_buyer"] = float(q.get("revenue", np.nan)) / r["buyers"] if r["buyers"] else np.nan
        if not tier.empty:
            t = tier[(tier["intention"] == k) & tier["price_tier"].isin(["premium", "luxury"])]
            r["premium_share"] = float(t["purchase_share_in_intention_pct"].sum()) if len(t) else np.nan
        if not prio.empty:
            m = prio[prio["topic"] == k]
            if len(m):
                r["freq_month"] = float(m["purchases_per_customer"].iloc[0])
                r["spend_month"] = float(m["spend_per_customer"].iloc[0])
                r["focus"] = str(m["focus"].iloc[0])
        r["price_index"] = r.get("avg_price", np.nan) / base * 100 if base == base and base > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)
