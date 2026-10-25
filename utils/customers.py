"""
customers.py — TAB 2: Customer preference analysis.
Who the customers of each topic are (age, membership), how often they buy, how much they spend, what price level they choose.
All numbers from the Kaggle outputs (PART 2 population tables, Section 9-10 files).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import kaggle_data as kd
from utils.data_loader import load_intention_labels
from utils.population import (population_ready, _hbar, _short, _not_ready_message,
                              _tab_demographics, _tab_price_tiers, _tab_frequency)
from utils.theme import intention_color, thin_rule
from utils.topicfacts import topic_facts
from utils.priority import priority_table, render_priority_view
from utils.data_loader import load_monthly_trends
from utils.trends import with_period_columns, aggregate_period
from utils.thesis_data import segment_static
from utils.ui import tile


def _answer(col, label, f, key, fmt, unit, largest=True):
    s = f[key].dropna()
    if s.empty:
        col.markdown(tile(label, "n/a"), unsafe_allow_html=True)
        return
    r = f.loc[s.idxmax() if largest else s.idxmin()]
    col.markdown(tile(label, f"T{int(r['topic'])} · {r['name']}", f"{fmt(r[key])} {unit}"), unsafe_allow_html=True)


def render_customers():
    st.markdown("### Who prefers what")
    st.caption("Customer preference by topic: which customers buy most, spend most, choose higher prices, and who they are.")
    if not population_ready():
        _not_ready_message()
        return
    f = topic_facts()
    labels = load_intention_labels()

    c = st.columns(3)
    _answer(c[0], "Buys most often", f, "purchases_per_buyer", lambda v: f"{v:.1f}", "purchases per buyer")
    _answer(c[1], "Spends most per customer", f, "spend_per_buyer", lambda v: f"{v:.3f}", "per buyer (normalised units)")
    _answer(c[2], "Prefers higher prices", f, "premium_share", lambda v: f"{v:.0f}%", "of purchases are premium or luxury tier")
    c = st.columns(3)
    _answer(c[0], "Most loyal customers", f, "loyal_pct", lambda v: f"{v:.0f}%", "have bought 10+ times")
    _answer(c[1], "Youngest customers", f, "mean_age", lambda v: f"{v:.1f}", "years on average", largest=False)
    _answer(c[2], "Oldest customers", f, "mean_age", lambda v: f"{v:.1f}", "years on average")

    thin_rule()
    st.markdown("#### Topic by topic")
    colors = [intention_color(int(k)) for k in f["topic"]]
    names = [f"T{int(k)} · {n}" for k, n in zip(f["topic"], f["name"])]
    g1, g2, g3 = st.columns(3)
    with g1:
        _hbar(names, f["purchases_per_buyer"].fillna(0).tolist(), colors, 380, "Purchases per buyer", "cu_ppb", suffix="", fmt="{:.1f}")
    with g2:
        _hbar(names, f["spend_per_buyer"].fillna(0).tolist(), colors, 380, "Spend per buyer", "cu_spb", suffix="", fmt="{:.3f}")
    with g3:
        _hbar(names, f["premium_share"].fillna(0).tolist(), colors, 380, "% of purchases in premium or luxury tier", "cu_prem", suffix="%", fmt="{:.0f}")
    g1, g2, g3 = st.columns(3)
    with g1:
        _hbar(names, f["loyal_pct"].fillna(0).tolist(), colors, 380, "% of customers with 10+ purchases", "cu_loyal", suffix="%", fmt="{:.0f}")
    with g2:
        _hbar(names, f["mean_age"].fillna(0).tolist(), colors, 380, "Average age", "cu_age", suffix=" yrs", fmt="{:.1f}")
    with g3:
        _hbar(names, f["price_index"].fillna(0).tolist(), colors, 380, "Average price paid (index, 100 = median)", "cu_pidx", suffix="", fmt="{:.0f}")

    show = f[["topic", "name", "customers", "customer_share", "purchases_per_buyer", "spend_per_buyer", "price_index", "premium_share", "loyal_pct", "mean_age"]].copy()
    show["topic"] = [f"T{int(k)}" for k in show["topic"]]
    show = show.rename(columns={"topic": "Topic", "name": "Preference", "customers": "Customers", "customer_share": "% of customers",
                                "purchases_per_buyer": "Purchases / buyer", "spend_per_buyer": "Spend / buyer", "price_index": "Price index",
                                "premium_share": "% premium+luxury", "loyal_pct": "% buying 10+", "mean_age": "Mean age"})
    st.dataframe(show, hide_index=True, use_container_width=True, column_config={
        "Customers": st.column_config.NumberColumn(format="%d"), "% of customers": st.column_config.NumberColumn(format="%.1f%%"),
        "Purchases / buyer": st.column_config.NumberColumn(format="%.1f"), "Spend / buyer": st.column_config.NumberColumn(format="%.3f"),
        "Price index": st.column_config.NumberColumn(format="%.0f"), "% premium+luxury": st.column_config.NumberColumn(format="%.0f%%"),
        "% buying 10+": st.column_config.NumberColumn(format="%.0f%%"), "Mean age": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Customers = active customers grouped by their dominant preference topic. Purchases, spend and price are per product topic "
               "bought. Spend is in the dataset's normalised price units. Source: population tables (PART 2), user and gap files (Sections 9-10).")

    thin_rule()
    st.markdown("#### Age, price and frequency")
    t_dem, t_price, t_freq = st.tabs(["Age and membership", "Price tiers", "How often they buy"])
    with t_dem:
        _tab_demographics(labels)
    with t_price:
        _tab_price_tiers(labels)
    with t_freq:
        _tab_frequency(labels)

    thin_rule()
    gap_by_k = {k: segment_static(k)["gap"] for k in range(10)}
    prio = priority_table(aggregate_period(with_period_columns(load_monthly_trends()), "All time"), "All time", None, gap_by_k)
    render_priority_view(prio, "all time", key_prefix="cu_prio")
