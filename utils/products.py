"""
products.py — TAB 3: Product and topic analysis.
One topic at a time: supply and demand, price, what defines it, best and slowest sellers with photos, product mix,
and what to add or reduce. Photos and sales statistics are real; the product mix is computed on the random 5%
stratified sample only (the other sample types are chosen, not random).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import kaggle_data as kd
from utils.catalog import products, top_products, has_sales
from utils.data_loader import load_monthly_trends
from utils.merch import action, action_chip
from utils.preference_labels import preference_name, preference_full_name
from utils.theme import intention_color, status_badge, thin_rule, THREAD, SIGNAL, MUTED
from utils.thesis_data import recommendation_text
from utils.topicfacts import topic_facts
from utils.trends import with_period_columns
from utils.ui import chip, tile, topic_title, product_cards

FONT = "IBM Plex Sans, sans-serif"


def _style(fig, h, title=None):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40 if title else 10, b=10), font=dict(family=FONT, size=13),
                      title=dict(text=title, font=dict(size=15), x=0) if title else None,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _mix_chart(df, col, title, k, key, top=10):
    vc = df[col].astype(str).value_counts().head(top).sort_values()
    if vc.empty:
        return
    fig = go.Figure(go.Bar(x=(vc / len(df) * 100).round(1), y=vc.index, orientation="h", marker_color=intention_color(k),
                           text=[f"{v / len(df) * 100:.0f}%" for v in vc.values], textposition="outside", cliponaxis=False))
    fig.update_xaxes(title="% of the topic's sampled products", gridcolor="#ECE9E1")
    _style(fig, 340, title)
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_products():
    f, prods = topic_facts(), products()
    st.markdown("### Products, topic by topic")
    st.caption("Pick a preference topic to see its supply and demand, what defines it, its real best and slowest sellers, and what to add or reduce.")
    k = st.selectbox("Preference topic", list(range(10)), format_func=lambda t: f"T{t} · {preference_full_name(t)}", key="prod_topic")
    r = f[f["topic"] == k].iloc[0]
    emoji, label, colour = status_badge(r["gap"])
    st.markdown(topic_title(k, r["line"]), unsafe_allow_html=True)

    c = st.columns(6)
    items = [
        ("Supply: share of range", f"{r['catalogue_share']:.1f}%", f"{kd.n(r['products'])} products"),
        ("Demand: share of purchases", f"{r['purchase_share']:.1f}%", f"{kd.n(r.get('purchases'))} items bought"),
        ("Gap", f"{r['gap']:+.1f}pp", f"{emoji} {label}"),
        ("Price index", f"{r['price_index']:.0f}" if r["price_index"] == r["price_index"] else "n/a", f"{r['price_tier']} tier" if r["price_tier"] else "100 = median price"),
        ("Buys per customer", f"{r['purchases_per_buyer']:.1f}" if r.get("purchases_per_buyer") == r.get("purchases_per_buyer") else "n/a", "purchases per buyer"),
        ("Spend per customer", f"{r['spend_per_buyer']:.3f}" if r.get("spend_per_buyer") == r.get("spend_per_buyer") else "n/a", "normalised price units"),
    ]
    for col, (a, b, d) in zip(c, items):
        col.markdown(tile(a, b, d), unsafe_allow_html=True)

    lab, why, col = action(r["gap"], float("nan"))
    st.markdown(f"**Brand intention for this topic:** {action_chip(lab if r['gap'] > 5 or r['gap'] < -5 else 'Balanced', col)} &nbsp; {recommendation_text(k)}",
                unsafe_allow_html=True)

    ka = kd.key_attributes()
    if not ka.empty:
        sub = ka[ka["topic"] == k].sort_values("rank").head(8)
        st.markdown("**What defines this preference** (the topic's strongest product features): " + " ".join(chip(f"{a_t}: {a}", "#EFEDE6", "#181A20") for a_t, a in zip(sub["attribute_type"], sub["attribute"])), unsafe_allow_html=True)

    thin_rule()
    pool = prods[prods["topic"] == k]
    if has_sales(pool):
        t1, t2, t3 = st.tabs(["Best sellers", "Most typical of the preference", "Slowest sellers"])
        with t1:
            product_cards(top_products(prods, k, 10, "bestsellers"))
        with t2:
            product_cards(top_products(prods, k, 10, "typical"))
        with t3:
            st.caption("From the random 5% sample, so a slow seller here is not an effect of how products were picked.")
            product_cards(top_products(prods, k, 10, "slowest"))
    else:
        product_cards(top_products(prods, k, 10, "typical"), show_action=False)

    thin_rule()
    samp = pool[pool["sample_type"].isin(["stratified", "unknown"])]
    st.markdown(f"#### Product mix of the topic ({len(samp):,} randomly sampled products)")
    m1, m2 = st.columns(2)
    with m1:
        _mix_chart(samp, "product_type_name", "Product types", k, "pr_type") if "product_type_name" in samp else None
    with m2:
        _mix_chart(samp, "colour_group_name", "Colours", k, "pr_col") if "colour_group_name" in samp else None
    m1, m2 = st.columns(2)
    with m1:
        if "avg_price" in samp:
            from utils.catalog import price_index
            idx = samp["avg_price"].map(price_index).dropna()
            if len(idx):
                fig = go.Figure(go.Histogram(x=idx.clip(upper=idx.quantile(0.98)), nbinsx=24, marker_color=intention_color(k)))
                fig.add_vline(x=100, line_dash="dash", line_color=MUTED)
                fig.update_xaxes(title="Price index (100 = median price paid)", gridcolor="#ECE9E1"); fig.update_yaxes(title="Products")
                _style(fig, 340, "How the topic's products are priced")
                st.plotly_chart(fig, use_container_width=True, key="pr_price")
    with m2:
        if has_sales(samp) and "avg_price" in samp:
            from utils.catalog import price_index
            d = samp.assign(pi=samp["avg_price"].map(price_index)).dropna(subset=["pi", "units"])
            if len(d):
                fig = go.Figure(go.Scatter(x=d["pi"].clip(upper=d["pi"].quantile(0.98)), y=d["units"], mode="markers",
                                           marker=dict(color=intention_color(k), size=7, opacity=0.55),
                                           text=d["prod_name"], hovertemplate="%{text}<br>price index %{x:.0f}<br>%{y:,} sold<extra></extra>"))
                fig.update_xaxes(title="Price index", gridcolor="#ECE9E1"); fig.update_yaxes(title="Units sold", type="log", gridcolor="#ECE9E1")
                _style(fig, 340, "Price against units sold (log scale)")
                st.plotly_chart(fig, use_container_width=True, key="pr_scatter")

    thin_rule()
    st.markdown("#### Supply and demand over time")
    mt = with_period_columns(load_monthly_trends())
    tot = mt.groupby("year_month")["units"].sum()
    sel = mt[mt["intention"] == k].set_index("year_month")["units"]
    share = (sel / tot * 100).dropna()
    fig = go.Figure(go.Scatter(x=share.index, y=share.values, mode="lines+markers", line=dict(color=intention_color(k), width=3), name="Share of purchases"))
    fig.add_hline(y=r["catalogue_share"], line_dash="dash", line_color=MUTED, annotation_text=f"Share of the range: {r['catalogue_share']:.1f}%")
    fig.update_yaxes(title="% of all purchases", gridcolor="#ECE9E1"); fig.update_xaxes(tickangle=-45)
    _style(fig, 340, "Above the dashed line = demand above the range")
    st.plotly_chart(fig, use_container_width=True, key="pr_time")
    st.caption("Source: monthly_segment_trends.csv (all transactions) and topic_gap_analysis.csv (Section 9).")
