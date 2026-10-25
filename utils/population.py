"""
population.py
================================================================================
Population-level ("All customers") analysis for the Customers tab.

All numbers come from PART 2 of kaggle/HM_KAGGLE_EXPORT_ALL.ipynb, which aggregates the
FULL 1.37M-customer / 31.8M-transaction dataset into a handful of tiny CSVs
(no customer IDs, no personal data) stored in  data/population/  inside data_export.zip. Nothing here is estimated from the 5% demo sample.

Definitions (also shown in the UI):
- "Customers by preference"  = active customers grouped by their DOMINANT preference
  (user_intention_weights.csv).
- "Purchases by preference"  = transactions grouped by the purchased article's
  dominant preference.  The two differ: a customer belongs to one preference, but
  buys across several.
- Price tier = global quartile of transaction price (thesis Section 10.3):
  Budget (Q1) < Q25 <= Medium (Q2) < Q50 <= Premium (Q3) < Q75 <= Luxury (Q4).
- Prices in this dataset are normalised, so "spending" is in normalised units,
  not currency.
================================================================================
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.theme import intention_color, thin_rule, THREAD, SIGNAL, MUTED

from utils.kaggle_data import population_dir


def _pop_dir() -> Path:
    """data/population (inside data_export.zip), else assets/population in the repo."""
    return Path(population_dir())

REQUIRED_FILES = [
    "population_summary.json", "pop_segments.csv", "pop_price_tier.csv",
    "pop_frequency.csv", "pop_age_groups.csv",
]

TIER_ORDER = ["budget", "medium", "premium", "luxury"]
TIER_LABEL = {"budget": "Budget (Q1)", "medium": "Medium (Q2)",
              "premium": "Premium (Q3)", "luxury": "Luxury (Q4)"}
TIER_COLOR = {"budget": "#CFD6EA", "medium": "#9AA6D6", "premium": "#5B5FA8", "luxury": SIGNAL}
AGE_ORDER = ["16-25", "26-35", "36-45", "46-55", "56-65", "66-80", "80+"]
FREQ_ORDER = ["1", "2-4", "5-9", "10-19", "20-49", "50+"]
FONT = "IBM Plex Sans, sans-serif"


# ----------------------------------------------------------------------------
# Loading (graceful when the files have not been generated yet)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_pop_table(name: str) -> pd.DataFrame:
    path = _pop_dir() / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_pop_summary() -> dict:
    path = _pop_dir() / "population_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def population_ready() -> bool:
    return all((_pop_dir() / f).exists() for f in REQUIRED_FILES)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _short(k: int, labels: dict) -> str:
    """'T1 · Ladieswear Upper Body' — the part of the name before the colon."""
    return f"T{k} · {labels[str(k)]['name'].split(':')[0]}"


def _style(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        title=dict(text=title, font=dict(size=15), x=0) if title else None,
        font=dict(family=FONT, size=13), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _hbar(labels, values, colors, height, title, key, suffix="%", fmt="{:.1f}", xmax=None):
    order = np.argsort(values)                      # plotly draws bottom-to-top
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colors = [colors[i] for i in order]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[fmt.format(v) + suffix for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:,.2f}" + suffix + "<extra></extra>",
    ))
    fig.update_xaxes(range=[0, (xmax or max(values)) * 1.32], showgrid=True, gridcolor="#ECE9E1", title=None)
    fig.update_yaxes(title=None)
    _style(fig, height, title)
    st.plotly_chart(fig, use_container_width=True, key=key)


def _heatmap(z, x, y, height, title, key, fmt="{:.0f}%", zmax=None):
    text = [[fmt.format(v) if pd.notna(v) else "" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, text=text, texttemplate="%{text}", zmin=0, zmax=zmax,
        colorscale=[[0, "#FAFAF8"], [1, THREAD]], showscale=False,
        hovertemplate="%{y} · %{x}<br>%{z:.1f}%<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title=None)
    _style(fig, height, title)
    st.plotly_chart(fig, use_container_width=True, key=key)


def _fmt_int(n) -> str:
    return f"{int(n):,}"


def _kpi(col, label: str, value: str, note: str | None = None):
    """Metric with a plain caption underneath (st.metric's delta shows a misleading arrow)."""
    with col:
        st.metric(label, value)
        if note:
            st.caption(note)


def _yes_no(v) -> str:
    try:
        return "Yes" if float(v) == 1 else "No"
    except (TypeError, ValueError):
        return "Unknown"


def _not_ready_message():
    st.warning(
        "📊 The population-level numbers have not been generated yet.\n\n"
        "1. Run **PART 2** of **`scripts/kaggle_all_in_one.py`** in the Kaggle notebook (set `RUN_PART_1 = False` to skip the sampling part).\n"
        "2. Run PART 5 (it puts the files inside `data_export.zip`), upload that zip to Google Drive "
        "and reboot the app on Streamlit Cloud."
    )


# ----------------------------------------------------------------------------
# Period-aware view (uses the monthly trends; works without the population files)
# ----------------------------------------------------------------------------
def render_period_view(seg_df: pd.DataFrame, period_label: str, key_prefix: str = "pop"):
    import plotly.express as px
    st.markdown(f"**Customers active per preference — {period_label}**")
    fig = px.bar(
        seg_df.sort_values("users"), x="users", y="label", orientation="h",
        color="label", color_discrete_map=dict(zip(seg_df["label"], seg_df["color"])),
        labels={"users": f"Customers active ({period_label})", "label": ""},
    )
    fig.update_layout(showlegend=False)
    _style(fig, 380)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_period_bar")
    st.caption(
        f"Counts are recomputed for **{period_label}** from real transactions "
        "(sidebar period selector). The charts in the other tabs are all-time."
    )


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
def _tab_segments(labels, seg_df, period_label):
    seg = load_pop_table("pop_segments").sort_values("intention").reset_index(drop=True)
    names = [_short(int(k), labels) for k in seg["intention"]]
    colors = [intention_color(int(k)) for k in seg["intention"]]

    c1, c2 = st.columns(2)
    with c1:
        _hbar(names, seg["customer_share_pct"].tolist(), colors, 420,
              "% of customers, by their dominant preference", "pop_cust_share")
    with c2:
        _hbar(names, seg["purchase_share_pct"].tolist(), colors, 420,
              "% of all purchases, by preference of the product", "pop_purchase_share")
    st.caption(
        "Left: each customer counted once, in the preference that dominates their profile. "
        "Right: every transaction counted once, in the preference of the product bought. "
        "They differ because customers buy across several preferences."
    )

    top_buy = seg.loc[seg["purchase_share_pct"].idxmax()]
    top_cust = seg.loc[seg["customer_share_pct"].idxmax()]
    st.markdown(
        f"- Most-purchased preference: **T{int(top_buy['intention'])}** "
        f"({top_buy['purchase_share_pct']:.1f}% of purchases).\n"
        f"- Largest customer segment: **T{int(top_cust['intention'])}** "
        f"({top_cust['customer_share_pct']:.1f}% of customers)."
    )

    thin_rule()
    st.subheader("Segment table")
    tbl = seg.copy()
    tbl.insert(0, "Segment", names)
    static = seg_df[["T", "confidence", "gap"]].rename(columns={"T": "intention"})
    tbl = tbl.merge(static, on="intention", how="left")
    for c in ("n_customers", "n_purchases", "customers_10plus"):
        if c in tbl.columns:
            tbl[c] = tbl[c].map(lambda v: f"{int(v):,}")
    cols = {
        "Segment": "Segment", "n_customers": "Customers", "customer_share_pct": "% cust.",
        "n_purchases": "Purchases", "purchase_share_pct": "% purch.",
        "revenue_share_pct": "% revenue", "customers_10plus": "10+ purch. (n)",
        "customers_10plus_pct": "10+ purch. (%)", "confidence": "Topic conc.",
        "gap": "Gap (pp)",
    }
    tbl = tbl[[c for c in cols if c in tbl.columns]].rename(columns=cols)
    st.dataframe(
        tbl, use_container_width=True, hide_index=True,
        column_config={
            "% cust.": st.column_config.NumberColumn(format="%.2f%%"),
            "% purch.": st.column_config.NumberColumn(format="%.2f%%"),
            "% revenue": st.column_config.NumberColumn(format="%.2f%%"),
            "10+ purch. (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Topic conc.": st.column_config.NumberColumn(format="%.4f"),
            "Gap (pp)": st.column_config.NumberColumn(format="%+.2f"),
        },
    )
    st.caption(
        "10+ purch. = customers with at least 10 purchases. Topic conc. = mean dominant-topic "
        "probability of the preference (topic_summary.csv). Gap = demand share minus catalogue share (topic_gap_analysis.csv)."
    )
    with st.expander("Customers active in the period selected in the sidebar"):
        render_period_view(seg_df, period_label)


def _tab_price_tiers(labels):
    pt = load_pop_table("pop_price_tier")
    summary = load_pop_summary()
    th = summary.get("price_thresholds", {})
    rows = [-1] + list(range(10))
    ylabels = ["All preferences"] + [_short(k, labels) for k in range(10)]

    def matrix(col):
        out = []
        for k in rows:
            sub = pt[pt["intention"] == k].set_index("price_tier")[col]
            out.append([float(sub.get(t, np.nan)) for t in TIER_ORDER])
        return np.array(out)

    share = matrix("purchase_share_in_intention_pct")
    reach = matrix("buyers_pct_of_intention_buyers")

    # --- stacked 100% bars: share of purchases by price tier ---
    fig = go.Figure()
    for j, t in enumerate(TIER_ORDER):
        vals = share[:, j]
        fig.add_trace(go.Bar(
            x=vals, y=ylabels, orientation="h", name=TIER_LABEL[t], marker_color=TIER_COLOR[t],
            text=[f"{v:.0f}%" if v >= 6 else "" for v in vals], textposition="inside",
            textfont=dict(color="#FFFFFF" if t in ("premium", "luxury") else "#181A20"),
            hovertemplate="%{y}<br>" + TIER_LABEL[t] + ": %{x:.1f}% of purchases<extra></extra>",
        ))
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.07, x=0, yanchor="top", traceorder="normal"))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title=None)
    fig.update_yaxes(autorange="reversed", title=None)
    _style(fig, 470, "Share of each preference's purchases, by price tier")
    st.plotly_chart(fig, use_container_width=True, key="pop_tier_stack")

    # --- reach heatmap: % of a preference's buyers who bought at least once in each tier ---
    _heatmap(reach, [TIER_LABEL[t] for t in TIER_ORDER], ylabels, 470,
             "% of each preference's buyers who bought at least once in a tier", "pop_tier_reach", zmax=100)
    st.caption(
        "A customer can buy in several tiers, so the rows of the heatmap add up to more than 100%. "
        "The bars above add up to 100% because they count purchases, not people."
    )

    lux = share[1:, TIER_ORDER.index("luxury")]
    bud = share[1:, TIER_ORDER.index("budget")]
    k_lux, k_bud = int(np.nanargmax(lux)), int(np.nanargmax(bud))
    st.markdown(
        f"- Most premium-leaning: **T{k_lux}** — {lux[k_lux]:.0f}% of its purchases are Luxury-tier.\n"
        f"- Most budget-leaning: **T{k_bud}** — {bud[k_bud]:.0f}% of its purchases are Budget-tier."
    )
    if th:
        st.caption(
            "Tiers are global quartiles of transaction price (thesis Section 10.3): "
            f"Budget < {th['q25']:.4f} ≤ Medium < {th['q50']:.4f} ≤ Premium < {th['q75']:.4f} ≤ Luxury. "
            "Prices are normalised by the dataset provider, so these are relative tiers, not currency bands."
        )


def _tab_frequency(labels):
    fq = load_pop_table("pop_frequency")
    summary = load_pop_summary()
    overall = fq[fq["intention"] == -1].set_index("freq_bin").reindex(FREQ_ORDER)
    total = int(overall["n_customers"].sum())
    n10 = int(overall.loc[["10-19", "20-49", "50+"], "n_customers"].sum())
    n20 = int(overall.loc[["20-49", "50+"], "n_customers"].sum())
    n50 = int(overall.loc["50+", "n_customers"])

    m1, m2, m3 = st.columns(3)
    _kpi(m1, "Customers with 10+ purchases", _fmt_int(n10), f"{n10 / total * 100:.1f}% of active customers")
    _kpi(m2, "Customers with 20+ purchases", _fmt_int(n20), f"{n20 / total * 100:.1f}% of active customers")
    _kpi(m3, "Customers with 50+ purchases", _fmt_int(n50), f"{n50 / total * 100:.1f}% of active customers")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=FREQ_ORDER, y=overall["share_pct"], marker_color=[THREAD if b in ("10-19", "20-49", "50+") else "#B8B4CF" for b in FREQ_ORDER],
            text=[f"{v:.1f}%" for v in overall["share_pct"]], textposition="outside", cliponaxis=False,
            customdata=overall["n_customers"],
            hovertemplate="%{x} purchases<br>%{customdata:,} customers<extra></extra>",
        ))
        fig.update_yaxes(range=[0, overall["share_pct"].max() * 1.2], ticksuffix="%", gridcolor="#ECE9E1")
        fig.update_xaxes(title="Purchases per customer")
        _style(fig, 400, "How many purchases does each customer have?")
        st.plotly_chart(fig, use_container_width=True, key="pop_freq_overall")
    with c2:
        rows = []
        for k in range(10):
            sub = fq[fq["intention"] == k].set_index("freq_bin").reindex(FREQ_ORDER)
            tot = sub["n_customers"].sum()
            hi = sub.loc[["10-19", "20-49", "50+"], "n_customers"].sum()
            rows.append((k, hi / tot * 100 if tot else 0.0))
        _hbar([_short(k, labels) for k, _ in rows], [v for _, v in rows],
              [intention_color(k) for k, _ in rows], 400,
              "% of each segment's customers with 10+ purchases", "pop_freq_by_intent")
    st.caption(
        f"Overall: {n10 / total * 100:.1f}% of {total:,} active customers have 10 or more purchases "
        "(bars in dark indigo on the left). Segment = the customer's dominant preference."
    )
    best = max(rows, key=lambda r: r[1])
    st.markdown(f"- Most loyal segment: **T{best[0]}** — {best[1]:.1f}% of its customers have 10+ purchases.")


def _tab_demographics(labels):
    ag = load_pop_table("pop_age_groups").set_index("age_group").reindex(AGE_ORDER).reset_index()
    summary = load_pop_summary()

    m1, m2, m3 = st.columns(3)
    _kpi(m1, "Median age", f"{summary.get('median_age', float('nan')):.0f}" if summary else "n/a", "years, active customers")
    top_c = ag.loc[ag["n_customers"].idxmax()]
    top_s = ag.loc[ag["total_spending"].idxmax()]
    _kpi(m2, "Largest age group (customers)", top_c["age_group"], f"{top_c['customers_share_pct']:.1f}% of customers")
    _kpi(m3, "Largest age group (spending)", top_s["age_group"], f"{top_s['spending_share_pct']:.1f}% of spending")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=ag["age_group"], y=ag["total_spending"], marker_color=THREAD,
            text=[f"{v:,.0f}" for v in ag["total_spending"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>%{y:,.0f} normalised units<extra></extra>",
        ))
        fig.update_yaxes(title="Total spending (normalised price units)", gridcolor="#ECE9E1")
        fig.update_xaxes(title="Age group")
        _style(fig, 400, "Total spending by age group")
        st.plotly_chart(fig, use_container_width=True, key="pop_age_spend")
    with c2:
        fig = go.Figure(go.Bar(
            x=ag["age_group"], y=ag["n_customers"], marker_color=SIGNAL,
            text=[f"{int(v):,}" for v in ag["n_customers"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>%{y:,} customers<extra></extra>",
        ))
        fig.update_yaxes(title="Number of customers", gridcolor="#ECE9E1")
        fig.update_xaxes(title="Age group")
        _style(fig, 400, "Number of customers by age group")
        st.plotly_chart(fig, use_container_width=True, key="pop_age_cust")
    st.caption("Prices in this dataset are normalised, so spending is shown in normalised units, not currency.")

    thin_rule()
    ai = load_pop_table("pop_age_topic")
    if not ai.empty:
        st.subheader("Which preferences does each age group buy?")
        basis = st.radio(
            "Measure", ["Share of the age group's purchases", "Share of the age group's customers (dominant preference)"],
            horizontal=True, key="pop_age_basis",
        )
        col = "purchase_share_in_age_pct" if basis.startswith("Share of the age group's purchases") else "customer_share_in_age_pct"
        ai = ai.assign(intention=ai["intention"].astype(int))
        piv = ai.pivot_table(index="age_group", columns="intention", values=col, aggfunc="sum").reindex(AGE_ORDER)
        piv = piv.reindex(columns=range(10))
        _heatmap(piv.values, [f"T{k}" for k in range(10)], list(piv.index), 380,
                 "Each row adds up to 100%", "pop_age_intent")
        dom = piv.idxmax(axis=1)
        st.markdown("- Dominant preference by age group: " + ", ".join(f"**{a}** → T{int(k)}" for a, k in dom.items()) + ".")
        thin_rule()

    mb = load_pop_table("pop_membership")
    if not mb.empty:
        st.subheader("Membership and communication")
        cols = st.columns(3)
        for col_ui, attr, title in zip(cols, ["club_member_status", "fashion_news_frequency", "Active"],
                                       ["Club member status", "Fashion news frequency", "Active flag"]):
            sub = mb[mb["attribute"] == attr].sort_values("share_pct", ascending=True).copy()
            if sub.empty:
                continue
            sub["value"] = sub["value"].map(_yes_no) if attr in ("Active", "FN") else sub["value"].astype(str)
            with col_ui:
                fig = go.Figure(go.Bar(
                    x=sub["share_pct"], y=sub["value"], orientation="h", marker_color=THREAD,
                    text=[f"{v:.1f}%" for v in sub["share_pct"]], textposition="outside", cliponaxis=False,
                ))
                fig.update_xaxes(range=[0, 118], ticksuffix="%", gridcolor="#ECE9E1")
                fig.update_yaxes(type="category", title=None)
                _style(fig, 260, title)
                st.plotly_chart(fig, use_container_width=True, key=f"pop_memb_{attr}")
        thin_rule()

    seg = load_pop_table("pop_segments").sort_values("intention")
    if "median_age" in seg.columns:
        st.subheader("Age profile by preference")
        _hbar([_short(int(k), labels) for k in seg["intention"]], seg["mean_age"].tolist(),
              [intention_color(int(k)) for k in seg["intention"]], 400,
              "Average age of each segment's customers", "pop_age_by_intent", suffix=" yrs", fmt="{:.1f}")


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def render_population_view(intention_labels: dict, seg_df: pd.DataFrame, period_label: str) -> bool:
    """Renders the whole 'All customers' analysis. Returns False (after showing
    set-up instructions and the period view) if the population files are missing."""
    if not population_ready():
        _not_ready_message()
        render_period_view(seg_df, period_label)
        return False

    s = load_pop_summary()
    seg = load_pop_table("pop_segments")
    top_buy = seg.loc[seg["purchase_share_pct"].idxmax()]
    k1, k2, k3, k4 = st.columns(4)
    _kpi(k1, "Active customers", _fmt_int(s["n_customers_active"]),
         f"of {_fmt_int(s['n_customers_registered'])} registered")
    _kpi(k2, "Customers with 10+ purchases", _fmt_int(s["n_customers_10plus"]),
         f"{s['pct_customers_10plus']:.1f}% of active customers")
    _kpi(k3, "Most-purchased preference", f"T{int(top_buy['intention'])}",
         f"{top_buy['purchase_share_pct']:.1f}% of all purchases")
    _kpi(k4, "Transactions analysed", _fmt_int(s["n_transactions"]), "full dataset, Sep 2018 – Sep 2020")

    t1, t2, t3, t4 = st.tabs(["Segments", "Price tiers", "Purchase frequency", "Demographics"])
    with t1:
        _tab_segments(intention_labels, seg_df, period_label)
    with t2:
        _tab_price_tiers(intention_labels)
    with t3:
        _tab_frequency(intention_labels)
    with t4:
        _tab_demographics(intention_labels)
    return True
