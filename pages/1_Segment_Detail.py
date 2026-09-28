"""
pages/1_Segment_Detail.py — Segment Detail
================================================================================
The drill-down for one intention segment, combining what used to be three
separate pages: a representative persona radar (from the old Recommendation
Demo), the product grid (from the old Product Catalog), and a real revenue
trend + rule-based recommendation (from the Performance Console concept) —
plus a "Recommendation Engine Audit" panel so the same screen also answers
"is the model treating this segment fairly?", not just "how is this segment
performing?".
================================================================================
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from utils.data_loader import (
    download_data, load_articles, load_feature_matrices, load_demo_personas,
    load_intention_labels, load_monthly_trends, model_paths, image_path,
)
from utils.models import load_models
from utils.recommender import score_catalog, explain_recommendation
from utils.theme import (
    inject_global_css, intention_color, intention_lens, render_sidebar_chrome,
    render_period_selector, status_badge, thin_rule, THREAD,
)
from utils.charts import intention_radar_chart, tower_contribution_chart
from utils.trends import with_period_columns, aggregate_period
from utils.thesis_data import segment_static, model_improvement, recommendation_text

st.set_page_config(page_title="Segment Detail", page_icon="🔍", layout="wide")
inject_global_css()
download_data()

render_sidebar_chrome()
monthly_df_raw = load_monthly_trends()
monthly_df = with_period_columns(monthly_df_raw)
granularity, period, compare_mode = render_period_selector(monthly_df_raw)

articles = load_articles()
personas = load_demo_personas()
intention_labels = load_intention_labels()

# ---- Segment selector (kept in sync with the Command Center) ----
default_k = st.session_state.get("selected_segment", 0)
options = [f"T{k} — {intention_labels[str(k)]['name']}" for k in range(10)]
choice = st.selectbox("Segment", options, index=default_k)
k = int(choice.split("—")[0].strip()[1:])
st.session_state["selected_segment"] = k

accent = intention_color(k)
static = segment_static(k)
emoji, status_label, status_color = status_badge(static["gap"])

st.markdown(
    f"""
    <div style="border-top:4px solid {accent}; padding-top:10px;">
        <div style="font-family:'Libre Caslon Display',serif; font-size:1.9rem;">
            T{k} — {intention_labels[str(k)]['name']}
        </div>
        <div style="font-style:italic; color:#8A8578;">{intention_lens(k)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers in segment", f"{static['users']:,}", f"{static['user_share']:.2f}% of base")
m2.metric("Avg. confidence", f"{static['confidence']:.4f}")
m3.metric("Supply-demand gap (txn-count, Table 5.2)", f"{static['gap']:+.2f}pp", status_label)
m4.metric("Model AUC gain (Table 4.7)", f"+{model_improvement(k):.1f}%")
st.caption(
    "The gap above is transaction-count based (Table 5.2, static). The "
    "revenue trend chart below is revenue-weighted and time-based — a "
    "different, complementary metric; the two are not directly comparable."
)

thin_rule()

# ============================================================================
# Radar (representative persona) + real monthly trend
# ============================================================================
radar_col, trend_col = st.columns(2)

with radar_col:
    st.subheader("Representative intention profile")
    seg_personas = personas[personas["dominant_intention"] == k]
    if len(seg_personas):
        rep = seg_personas.sort_values("confidence", ascending=False).iloc[0]
        vec = rep[[f"intention_{i}" for i in range(10)]].values.astype(np.float32)
        st.plotly_chart(
            intention_radar_chart(vec, intention_labels),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.caption(f"Persona confidence: {rep['confidence']:.2f} · {int(rep['n_purchases'])} historical purchases")
    else:
        st.info("No demo persona available for this segment.")

with trend_col:
    st.subheader("Revenue share trend (real transaction data)")
    seg_monthly = monthly_df[monthly_df["intention"] == k].sort_values("year_month")
    if len(seg_monthly):
        fig = px.line(
            seg_monthly, x="year_month", y="revenue_share_pct", markers=True,
            labels={"year_month": "Month", "revenue_share_pct": "Revenue share (%)"},
        )
        fig.update_traces(line_color=accent, marker_color=accent)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Share of total monthly revenue attributable to this segment, "
                   "computed from transaction dates (t_dat) — not simulated.")
    else:
        st.info("monthly_segment_trends.csv not found yet — run the Script 01 bonus step.")

thin_rule()

# ============================================================================
# Business recommendation (rule-based, same logic as thesis Section 5.4)
# ============================================================================
st.subheader("Recommended action")
st.markdown(f"{emoji} **{recommendation_text(k)}**")

thin_rule()

# ============================================================================
# Product grid for this segment
# ============================================================================
st.subheader("Representative products")
seg_articles = articles[articles["dominant_intention"] == k].head(12)
cols = st.columns(6)
for i, (_, product) in enumerate(seg_articles.iterrows()):
    with cols[i % 6]:
        ip = image_path(product["article_id"])
        if os.path.exists(ip):
            st.image(ip, use_container_width=True)
        st.caption(str(product.get("prod_name", ""))[:28])

thin_rule()

# ============================================================================
# Recommendation Engine Audit — is the model treating this segment fairly?
# ============================================================================
st.subheader("Recommendation engine audit")
st.caption(
    "Pick a real persona from this segment and inspect exactly what the "
    "Three-Tower model recommends and why — a QA check on the engine's "
    "behaviour, not a shopping demo."
)

seg_personas = personas[personas["dominant_intention"] == k]
if len(seg_personas) == 0:
    st.warning("No persona available for this segment to audit.")
else:
    persona_label = st.selectbox("Persona", seg_personas["persona_label"].tolist(), key=f"audit_persona_{k}")
    persona_row = seg_personas[seg_personas["persona_label"] == persona_label].iloc[0]

    if st.button("Run audit", type="primary"):
        three_path, two_path = model_paths()
        three_model, two_model = load_models(three_path, two_path)
        visual_feat, semantic_feat, art_feat_idx = load_feature_matrices()

        user_intention = persona_row[[f"intention_{i}" for i in range(10)]].values.astype(np.float32)
        user_demo = np.array([
            persona_row.get("age", 30.0), persona_row.get("FN", 0.0), persona_row.get("Active", 0.0),
        ], dtype=np.float32)

        with st.spinner("Scoring catalogue..."):
            top, _ = score_catalog(
                three_model, two_model, visual_feat, semantic_feat, art_feat_idx,
                articles, user_intention, user_demo, top_n=8,
            )
        st.session_state[f"audit_result_{k}"] = top

    if f"audit_result_{k}" in st.session_state:
        top = st.session_state[f"audit_result_{k}"]
        st.markdown(
            f"Avg. Three-Tower score **{top['three_tower_score'].mean():.3f}** vs. "
            f"Two-Tower **{top['two_tower_score'].mean():.3f}** "
            f"(Δ {top['score_delta'].mean():+.3f}) for this persona."
        )
        for _, product in top.iterrows():
            img_col, text_col, chart_col = st.columns([1, 2, 2])
            with img_col:
                ip = image_path(product["article_id"])
                if os.path.exists(ip):
                    st.image(ip, use_container_width=True)
            with text_col:
                st.markdown(f"**{str(product.get('prod_name', 'Product'))[:38]}**")
                st.caption(f"3T {product['three_tower_score']:.3f} · 2T {product['two_tower_score']:.3f}")
                st.markdown(explain_recommendation(product, intention_labels))
            with chart_col:
                tf = tower_contribution_chart(product["tower1_mag"], product["tower2_mag"], product["tower3_mag"])
                st.plotly_chart(tf, use_container_width=True, config={"displayModeBar": False})
            thin_rule()
