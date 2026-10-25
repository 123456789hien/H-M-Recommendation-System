"""
thesis_data.py
================================================================================
Per-topic figures used across the pages. Nothing is typed in: every value is
read from the Kaggle outputs through utils/kaggle_data.py.

  customers (active) ..... population/pop_segments.csv (Kaggle PART 2)
                           fallback: user_dominant_topic_dist.csv (Section 10; includes new customers)
  catalogue share ........ topic_gap_analysis.csv        (Section 9)
  demand (purchase) share  topic_gap_analysis.csv        (Section 9)
  supply-demand gap ...... topic_gap_analysis.csv        (Section 9)
  topic concentration .... topic_summary.csv             (Section 8)
  model improvement ...... per_topic_metrics.csv         (Section 14)
================================================================================
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

from utils import kaggle_data as kd


@st.cache_data(show_spinner=False)
def _static_table() -> dict:
    gap, summ, dist = kd.gap(), kd.topic_summary(), kd.user_dist()
    pop_path = os.path.join(kd.population_dir(), "pop_segments.csv")
    users = {}
    if os.path.exists(pop_path):
        seg = pd.read_csv(pop_path)
        users = {int(r.intention): int(r.n_customers) for r in seg.itertuples()}
    elif not dist.empty:
        users = {int(r.topic): int(r.user_count) for r in dist.itertuples()}
    total = sum(users.values()) or 1
    out = {}
    for k in range(10):
        g = gap[gap["topic"] == k]
        s = summ[summ["topic"] == k]
        out[k] = {
            "users": users.get(k, 0),
            "user_share": users.get(k, 0) / total * 100,
            "confidence": float(s["mean_dominant_prob"].iloc[0]) if len(s) else float("nan"),
            "cat_share": float(g["cat_share"].iloc[0]) * 100 if len(g) else float("nan"),
            "demand_share": float(g["user_share"].iloc[0]) * 100 if len(g) else float("nan"),
            "gap": float(g["gap_pp"].iloc[0]) if len(g) else float("nan"),
            "mean_price": float(s["mean_price"].iloc[0]) if len(s) else float("nan"),
            "article_count": int(s["article_count"].iloc[0]) if len(s) else 0,
        }
    return out


@st.cache_data(show_spinner=False)
def _improvement_table() -> dict:
    pt = kd.per_topic()
    if pt.empty:
        return {}
    return {int(r.intention): kd.rel_gain(r.three_tower_auc, r.two_tower_auc) for r in pt.itertuples()}


def segment_static(k: int) -> dict:
    return _static_table().get(int(k), {})


def model_improvement(k: int) -> float:
    """Relative AUC gain of the Three-Tower over the Two-Tower for products of topic k (%)."""
    return _improvement_table().get(int(k), float("nan"))


def recommendation_text(k: int) -> str:
    """Rule-based recommendation for a topic (product line), from the figures above."""
    s = segment_static(k)
    gap = s["gap"]
    conf = s["confidence"]
    share = s["user_share"]

    if gap > 5:
        supply_msg = f"Undersupplied by {gap:+.2f}pp — increase SKU allocation by 10-15%."
    elif gap < -5:
        supply_msg = f"Oversupplied by {gap:.2f}pp — rationalise SKUs by 10-15%."
    else:
        supply_msg = f"Supply roughly matches demand ({gap:+.2f}pp) — maintain current allocation."

    if conf > 0.65 and share < 2:
        invest_msg = "A clearly defined preference in a small customer group — a good candidate for automated, low-risk lifecycle marketing."
    elif conf < 0.55:
        invest_msg = "Products in this line are less clearly tied to one preference — the lighter Two-Tower model is enough here and saves inference cost; expected personalisation return is low."
    else:
        invest_msg = "A moderately clear preference — standard personalisation investment is appropriate."

    return f"{supply_msg} {invest_msg}"
