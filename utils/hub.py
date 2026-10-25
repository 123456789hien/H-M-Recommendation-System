"""
hub.py — TAB 1: Preference hub.
Every key figure of each of the 10 preference topics, price included, with 10 real products per topic.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from utils import kaggle_data as kd
from utils.catalog import products, top_products, caption, has_sales
from utils.data_loader import image_path
from utils.preference_labels import preference_name
from utils.theme import intention_color, status_badge
from utils.topicfacts import topic_facts
from utils.ui import chip, tile, topic_title


def _fmt(v, d=2, suffix=""):
    return f"{v:.{d}f}{suffix}" if v == v and v is not None else "n/a"


def render_hub():
    f, prods = topic_facts(), products()
    us, ov = kd.user_stats(), kd.overview()

    st.markdown("### What customers prefer, topic by topic")
    st.caption("10 preference topics found from product colour, pattern, material and description, category and price. "
               "For each: who buys it, how often, how much they spend, what it costs, and what the products look like.")

    def top(col, fmt):
        r = f.loc[f[col].idxmax()]
        return f"T{int(r['topic'])} · {r['name']}", fmt(r[col])

    c = st.columns(5)
    n1, v1 = top("purchase_share", lambda v: f"{v:.1f}% of all purchases")
    n2, v2 = top("purchases_per_buyer", lambda v: f"{v:.1f} purchases per buyer") if "purchases_per_buyer" in f else ("n/a", "")
    n3, v3 = top("spend_per_buyer", lambda v: f"{v:.3f} spent per buyer") if "spend_per_buyer" in f else ("n/a", "")
    n4, v4 = top("premium_share", lambda v: f"{v:.0f}% premium or luxury purchases") if "premium_share" in f else ("n/a", "")
    n5, v5 = top("customer_share", lambda v: f"{v:.1f}% of customers")
    for col, (lab, name, note) in zip(c, [("Most purchased", n1, v1), ("Buys most often", n2, v2), ("Spends most", n3, v3),
                                          ("Prefers higher prices", n4, v4), ("Largest customer base", n5, v5)]):
        col.markdown(tile(lab, name, note), unsafe_allow_html=True)
    st.caption(f"{kd.n(us.get('active_customers'))} active customers · {kd.n(ov.get('n_articles'))} products · "
               f"{kd.n(ov.get('n_transactions'))} transactions · price index 100 = median price paid (prices are normalised, not currency)")

    modes = ["Most typical of the preference"] + (["Best sellers"] if has_sales(prods) else [])
    mode = st.radio("10 products shown for each topic", modes, horizontal=True, key="hub_mode")
    by = "bestsellers" if mode == "Best sellers" else "typical"

    for _, r in f.iterrows():
        k = int(r["topic"])
        emoji, label, colour = status_badge(r["gap"])
        tier = f" · {r['price_tier']} price tier" if r["price_tier"] else ""
        st.markdown(topic_title(k, f"{r['line']}"), unsafe_allow_html=True)
        st.markdown(chip(f"{emoji} {label}: purchases {r['purchase_share']:.1f}% vs range {r['catalogue_share']:.1f}% ({r['gap']:+.1f}pp)", colour)
                    + ("&nbsp;" + chip(f"Price index {_fmt(r.get('price_index'), 0)}{tier}", "#5B5FA8") if r.get("price_index") == r.get("price_index") else "")
                    + (f"&nbsp;" + chip(f"Focus: {r['focus']}", "#2E8B57") if isinstance(r.get("focus"), str) else ""), unsafe_allow_html=True)
        cols = st.columns(6)
        for col, (lab, val, note) in zip(cols, [
            ("Customers", f"{r['customer_share']:.1f}%", f"{kd.n(r['customers'])} customers"),
            ("Purchases", f"{r['purchase_share']:.1f}%", f"{kd.n(r.get('purchases'))} items"),
            ("Revenue", f"{_fmt(r.get('revenue_share'), 1)}%", "share of all revenue"),
            ("Buys per customer", _fmt(r.get("purchases_per_buyer"), 1), "purchases per buyer, all time"),
            ("Spend per customer", _fmt(r.get("spend_per_buyer"), 3), "normalised price units"),
            ("Buyers 10+ times", f"{_fmt(r.get('loyal_pct'), 0)}%", "of the topic's customers"),
        ]):
            col.markdown(tile(lab, val, note), unsafe_allow_html=True)
        pick = top_products(prods, k, 10, by)
        if pick.empty:
            st.caption("No product photos for this topic in the demo sample.")
            continue
        paths = [image_path(a) for a in pick["article_id"]]
        caps = [caption(row, with_sales=(by == "bestsellers")) for _, row in pick.iterrows()]
        st.image(paths, width=104, caption=caps)
    st.caption("Photos: the real products of each topic in the demo sample. 'Most typical' = products carrying the most of the topic's "
               "defining features (Section 7-8); 'Best sellers' = most units sold over the whole dataset.")
