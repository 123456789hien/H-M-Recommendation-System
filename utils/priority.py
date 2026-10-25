"""
priority.py
================================================================================
"Where should the business focus?"  How often customers buy and how much they
spend, per preference topic (product line), for the period selected in the sidebar.

Computed from the real monthly table (monthly_segment_trends.csv, 100% of the
transactions): units, revenue and active customers per topic per month.

  purchases per customer = units / active customers     (buying frequency)
  spend per customer     = revenue / active customers   (spending budget)
  price per item         = revenue / units

"Active customers" are summed over months, so both ratios read "per active
customer per month", the same unit for month, quarter, year and all-time views.
Spend is in the dataset's normalised price units, not currency.

Focus label = position against the median of the 10 topics on both measures.
It is a rule of thumb for deciding where to look first, not a forecast.
================================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.theme import intention_color, THREAD, MUTED, thin_rule
from utils.preference_labels import preference_name, product_line

FOCUS_RULES = {
    (True, True): ("Invest & promote", "Customers buy often and spend more here."),
    (False, True): ("Grow repeat purchases", "Customers spend well but come back less often."),
    (True, False): ("Raise basket value", "Customers buy often but spend less each time."),
    (False, False): ("Maintain & review range", "Lower frequency and lower spend than most lines."),
}


def priority_table(agg: pd.DataFrame, granularity: str, period, gap_by_k: dict) -> pd.DataFrame:
    """One row per preference topic for the selected period."""
    if agg is None or agg.empty:
        return pd.DataFrame()
    if granularity == "All time":
        sub = agg
    else:
        from utils.trends import GRANULARITY_COL
        sub = agg[agg[GRANULARITY_COL[granularity]] == period]
    sub = sub[sub["n_customers"] > 0].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["topic"] = sub["intention"].astype(int)
    sub["purchases_per_customer"] = sub["units"] / sub["n_customers"]
    sub["spend_per_customer"] = sub["revenue"] / sub["n_customers"]
    sub["price_per_item"] = sub["revenue"] / sub["units"].replace(0, np.nan)
    sub["gap_pp"] = sub["topic"].map(gap_by_k)
    if len(sub) >= 4:
        hf = sub["purchases_per_customer"] >= sub["purchases_per_customer"].median()
        hs = sub["spend_per_customer"] >= sub["spend_per_customer"].median()
        sub["focus"] = [FOCUS_RULES[(bool(a), bool(b))][0] for a, b in zip(hf, hs)]
    else:
        sub["focus"] = "n/a"
    return sub.sort_values("topic").reset_index(drop=True)


def _ratio_sentence(a_name, b_name, a_val, b_val, what):
    if not b_val or not np.isfinite(a_val) or not np.isfinite(b_val):
        return None
    r = a_val / b_val
    if abs(r - 1) < 0.03:
        return f"**{a_name}** and **{b_name}** are about the same on {what}."
    more = "more" if r > 1 else "less"
    x = r if r > 1 else 1 / r
    return f"**{a_name}** customers {what} **{x:.1f}× {more}** than **{b_name}** customers."


def render_priority_view(tbl: pd.DataFrame, period_label: str, key_prefix: str = "prio"):
    st.subheader("Where should the business focus?")
    st.caption(
        "How often customers buy and how much they spend in each product line, for "
        f"**{period_label}**. Frequency = purchases per active customer per month; spend = money "
        "per active customer per month (normalised price units, not currency)."
    )
    if tbl.empty:
        st.info("No purchases in the selected period.")
        return

    names = {int(r.topic): f"T{int(r.topic)} · {preference_name(int(r.topic))}" for r in tbl.itertuples()}
    mf, ms = tbl["purchases_per_customer"].median(), tbl["spend_per_customer"].median()

    c1, c2 = st.columns([1.1, 1])
    with c1:
        size = np.sqrt(tbl["revenue"] / tbl["revenue"].max()) * 46 + 12
        fig = go.Figure(go.Scatter(
            x=tbl["purchases_per_customer"], y=tbl["spend_per_customer"], mode="markers+text",
            text=[f"T{t}" for t in tbl["topic"]], textposition="top center",
            textfont=dict(color="#181A20", size=12),
            marker=dict(size=size, color=[intention_color(t) for t in tbl["topic"]], line=dict(width=1, color="#FFFFFF")),
            customdata=np.stack([[names[int(t)] for t in tbl["topic"]], tbl["revenue_share_pct"], tbl["focus"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>Purchases per customer: %{x:.2f}<br>"
                          "Spend per customer: %{y:.4f}<br>Revenue share: %{customdata[1]:.1f}%<br>"
                          "%{customdata[2]}<extra></extra>",
        ))
        fig.add_vline(x=mf, line_dash="dash", line_color=MUTED, opacity=0.6)
        fig.add_hline(y=ms, line_dash="dash", line_color=MUTED, opacity=0.6)
        fig.update_xaxes(title="Purchases per active customer per month  →  buys more often", gridcolor="#ECE9E1")
        fig.update_yaxes(title="Spend per active customer per month  →  spends more", gridcolor="#ECE9E1")
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="IBM Plex Sans, sans-serif", size=13))
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_scatter")
        st.caption("Bubble size = share of revenue. Dashed lines = median of the 10 product lines. "
                   "Top-right = customers buy often and spend more.")
    with c2:
        show = tbl.assign(Topic=[names[int(t)] for t in tbl["topic"]])[
            ["Topic", "purchases_per_customer", "spend_per_customer", "focus"]
        ].rename(columns={
            "purchases_per_customer": "Purchases / customer", "spend_per_customer": "Spend / customer",
            "focus": "Suggested focus",
        }).sort_values("Spend / customer", ascending=False)
        st.dataframe(show, use_container_width=True, hide_index=True, height=35 * (len(show) + 1) + 3, column_config={
            "Purchases / customer": st.column_config.NumberColumn(format="%.2f"),
            "Spend / customer": st.column_config.NumberColumn(format="%.4f"),
        })
        st.caption("Sorted by spend per customer. Hover a bubble for revenue share.")

    top_f = tbl.loc[tbl["purchases_per_customer"].idxmax()]
    top_s = tbl.loc[tbl["spend_per_customer"].idxmax()]
    invest = [names[int(t)] for t, f in zip(tbl["topic"], tbl["focus"]) if f == "Invest & promote"]
    lines = [
        f"- Customers buy **most often** in **{names[int(top_f.topic)]}** ({top_f.purchases_per_customer:.2f} purchases per customer per month).",
        f"- Customers **spend most** in **{names[int(top_s.topic)]}** ({top_s.spend_per_customer:.4f} per customer per month).",
    ]
    if invest:
        lines.append("- Buy often **and** spend more: " + "; ".join(invest) + ". These are the first lines to prioritise for marketing and range investment.")
    st.markdown("\n".join(lines))

    with st.expander("Compare two product lines"):
        topics = [int(t) for t in tbl["topic"]]
        d1 = 5 if 5 in topics else topics[0]
        d2 = 1 if 1 in topics else topics[-1]
        ca, cb = st.columns(2)
        a = ca.selectbox("Product line A", topics, index=topics.index(d1), format_func=lambda t: names[t], key=f"{key_prefix}_a")
        b = cb.selectbox("Product line B", topics, index=topics.index(d2), format_func=lambda t: names[t], key=f"{key_prefix}_b")
        ra, rb = tbl[tbl["topic"] == a].iloc[0], tbl[tbl["topic"] == b].iloc[0]
        if a == b:
            st.caption("Pick two different product lines.")
        else:
            s1 = _ratio_sentence(preference_name(a), preference_name(b), ra.purchases_per_customer, rb.purchases_per_customer, "buy")
            s2 = _ratio_sentence(preference_name(a), preference_name(b), ra.spend_per_customer, rb.spend_per_customer, "spend")
            st.markdown("\n".join(f"- {s}" for s in (s1, s2) if s))
            st.caption("Computed from real transactions for the selected period. "
                       "Product lines differ in price level, so read spend together with price per item.")
