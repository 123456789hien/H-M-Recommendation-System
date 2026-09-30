"""
pages/2_Personalized_Experience.py — Personalized Shopping Experience
================================================================================
This is the page that answers "how does the Three-Tower architecture
actually apply to fashion retail, in a way a customer would feel?" — the
Command Center and Segment Detail pages report on the business; the
Recommendation Audit page inspects the model's technical behaviour; THIS
page shows the literal storefront a real shopper would see, side-by-side
under Two-Tower vs Three-Tower, translated into a metric anyone can read at
a glance: what fraction of the feed actually matches the shopper's own
dominant shopping intention.
================================================================================
"""

import os
import numpy as np
import streamlit as st

from utils.data_loader import (
    download_data, load_articles, load_feature_matrices, load_demo_personas,
    load_intention_labels, load_sampling_report, model_paths, image_path,
)
from utils.models import load_models
from utils.recommender import score_catalog
from utils.theme import inject_global_css, intention_color, intention_lens, render_sidebar_chrome, thin_rule

st.set_page_config(page_title="Personalized Shopping Experience", page_icon="✨", layout="wide")
inject_global_css()
download_data()
render_sidebar_chrome()

st.title("Personalized Shopping Experience")
st.caption(
    "This is the practical application of the Three-Tower architecture: a "
    "real storefront feed for a real customer, compared side-by-side against "
    "what the Two-Tower baseline would show the same customer. The other "
    "pages report on business trends or audit model internals — this page "
    "shows the actual shopping experience the technology creates."
)

articles = load_articles()
personas = load_demo_personas()
intention_labels = load_intention_labels()
report = load_sampling_report()
three_path, two_path = model_paths()
three_model, two_model = load_models(three_path, two_path)
intention_cols = [f"intention_{k}" for k in range(10)]

# ============================================================================
# Choose a shopper
# ============================================================================
st.header("Step 1 — Choose a shopper")

mode = st.radio("Mode", ["Real customer persona", "New customer (cold-start)"], horizontal=True)

if mode.startswith("Real"):
    intent_filter = st.selectbox(
        "Filter personas by intention",
        ["All intentions"] + [f"T{k} — {intention_labels[str(k)]['name']}" for k in range(10)],
    )
    if intent_filter == "All intentions":
        filtered_personas = personas
    else:
        fk = int(intent_filter.split("—")[0].strip()[1:])
        filtered_personas = personas[personas["dominant_intention"] == fk]

    persona_choice = st.selectbox("Persona", filtered_personas["persona_label"].tolist())
    persona_row = filtered_personas[filtered_personas["persona_label"] == persona_choice].iloc[0]
    user_intention = persona_row[intention_cols].values.astype(np.float32)
    age = float(persona_row.get("age", 30.0))
    fn = float(persona_row.get("FN", 0.0))
    active = float(persona_row.get("Active", 0.0))
    own_k = int(persona_row["dominant_intention"])
else:
    archetype_choice = st.selectbox(
        "Which segment does this shopper most identify with?",
        ["No preference — global prior only"] +
        [f"T{k} — {intention_labels[str(k)]['name']}" for k in range(10)],
    )
    global_prior = np.array([report["global_prior"][f"intention_{k}"] for k in range(10)], dtype=np.float32)
    if archetype_choice.startswith("No preference"):
        user_intention = global_prior
        own_k = int(np.argmax(global_prior))
    else:
        own_k = int(archetype_choice.split("—")[0].strip()[1:])
        boosted = global_prior.copy()
        boosted[own_k] += 0.3
        user_intention = boosted / boosted.sum()
    age, fn, active = 30.0, 0.0, 1.0

accent = intention_color(own_k)
st.markdown(
    f"<div style='border-left:4px solid {accent}; padding:8px 14px; background:#F1EFE9;'>"
    f"Shopping for: <b>T{own_k} — {intention_labels[str(own_k)]['name']}</b> · "
    f"<i>{intention_lens(own_k)}</i></div>",
    unsafe_allow_html=True,
)

thin_rule()

# ============================================================================
# Generate both storefronts
# ============================================================================
st.header("Step 2 — Compare the two storefronts")
n_products = st.slider("Feed size", 4, 12, 8, step=4)

if st.button("Generate storefronts", type="primary"):
    visual_feat, semantic_feat, art_feat_idx = load_feature_matrices()
    user_demo = np.array([age, fn, active], dtype=np.float32)
    with st.spinner("Scoring catalogue with both models..."):
        _, full_scored = score_catalog(
            three_model, two_model, visual_feat, semantic_feat, art_feat_idx,
            articles, user_intention, user_demo, top_n=n_products,
        )
    st.session_state["exp_full"] = full_scored
    st.session_state["exp_own_k"] = own_k

if "exp_full" in st.session_state and st.session_state.get("exp_own_k") == own_k:
    full_scored = st.session_state["exp_full"]
    three_feed = full_scored.sort_values("three_tower_score", ascending=False).head(n_products)
    two_feed = full_scored.sort_values("two_tower_score", ascending=False).head(n_products)

    three_match = (three_feed["dominant_intention"] == own_k).mean() * 100
    two_match = (two_feed["dominant_intention"] == own_k).mean() * 100

    thin_rule()
    st.subheader("The metric that matters to a shopper: does the feed match why I actually shop?")
    mc1, mc2 = st.columns(2)
    mc1.metric("Two-Tower feed — matches shopper's intention", f"{two_match:.0f}%")
    mc2.metric("Three-Tower feed — matches shopper's intention", f"{three_match:.0f}%",
               f"{three_match - two_match:+.0f}pp vs Two-Tower")
    st.caption(
        "Computed directly: the % of the feed whose own dominant intention "
        "matches this shopper's dominant intention (T"
        f"{own_k} — {intention_labels[str(own_k)]['name']}). This is an "
        "intuitive, customer-facing relevance measure — distinct from AUC, "
        "which measures ranking quality across the whole test set rather "
        "than feed-level topical relevance for one shopper."
    )

    thin_rule()
    feed_col1, feed_col2 = st.columns(2)

    def render_feed(container, feed_df, label):
        with container:
            st.markdown(f"#### {label}")
            cols = st.columns(2)
            for i, (_, product) in enumerate(feed_df.iterrows()):
                with cols[i % 2]:
                    ip = image_path(product["article_id"])
                    if os.path.exists(ip):
                        st.image(ip, use_container_width=True)
                    matches = int(product["dominant_intention"]) == own_k
                    tag = "✅ matches intent" if matches else "· different intent"
                    st.caption(f"{str(product.get('prod_name',''))[:26]}  \n{tag}")

    render_feed(feed_col1, two_feed, "Two-Tower storefront (baseline)")
    render_feed(feed_col2, three_feed, "Three-Tower storefront (this thesis)")
else:
    st.info("Choose a shopper above, then click **Generate storefronts**.")

thin_rule()
st.caption(
    "This page operationalises the thesis's central claim into a concrete "
    "fashion-retail application: Intention Alignment (Tower 3) is designed "
    "to increase the share of a customer's feed that matches their actual "
    "shopping motivation — a direct, felt improvement in relevance, on top "
    "of the aggregate AUC/Recall gains reported elsewhere in this console."
)
