"""
studio.py — TAB 4: Recommendation studio.
Choose any product, choose which customer preference to serve, and the Three-Tower model recommends similar products,
showing its match probability, the share of each tower, how well each product sells, topic supply and demand, and what to add or reduce.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import kaggle_data as kd
from utils.catalog import products, has_sales, price_index
from utils.data_loader import (image_path, load_customer_validation, load_feature_matrices, load_sampling_report, model_paths)
from utils.merch import action, action_chip, seller_label
from utils.models import load_models
from utils.preference_labels import preference_name, preference_full_name
from utils.studio_scoring import (typical_customer, one_customer, neutral_inputs, similar_ids, score_items)
from utils.theme import intention_color, status_badge, thin_rule, THREAD, SIGNAL, MUTED
from utils.thesis_data import segment_static
from utils.ui import chip, tile, topic_title, product_cards

FONT = "IBM Plex Sans, sans-serif"
TOWER_COL = {"visual": "#5B5FA8", "text": "#C9932E", "preference": "#2E8B57"}


def _towers_bar(row, key, h=120):
    fig = go.Figure()
    for nm, lab in (("visual", "Image (Tower 1)"), ("text", "Text and customer (Tower 2)"), ("preference", "Preference (Tower 3)")):
        fig.add_trace(go.Bar(x=[row[f"share_{nm}"]], y=[""], orientation="h", name=lab, marker_color=TOWER_COL[nm],
                             text=[f"{row[f'share_{nm}']:.0f}%"], textposition="inside", textfont=dict(color="#fff", size=14),
                             hovertemplate=f"{lab}<br>{row[f'share_{nm}']:.1f}% of the towers' effect<br>adds {row[f'drop_{nm}']:+.1f} points of probability<extra></extra>"))
    fig.update_layout(barmode="stack", height=h, margin=dict(l=0, r=0, t=0, b=0), font=dict(family=FONT, size=13),
                      legend=dict(orientation="h", y=-0.35, x=0), xaxis=dict(range=[0, 100], visible=False), yaxis=dict(visible=False),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key=key)


def _rec_extra(row):
    return (f"<div style='margin-top:6px; padding:6px 8px; background:#F4F2EC; border-radius:6px; font-size:0.82rem;'>"
            f"<b>Three-Tower match {row['three_p'] * 100:.1f}%</b><br>"
            f"Two-Tower {row['two_p'] * 100:.1f}% · uplift {row['uplift_pp']:+.1f}pp<br>"
            f"<span style='color:#5B5FA8'>Image {row['share_visual']:.0f}%</span> · "
            f"<span style='color:#C9932E'>Text {row['share_text']:.0f}%</span> · "
            f"<span style='color:#2E8B57'>Preference {row['share_preference']:.0f}%</span><br>"
            f"<span style='color:#8A8578'>Similarity to your product {row['similarity'] * 100:.0f}%</span></div>")


def render_studio():
    prods = products()
    st.markdown("### Recommendation studio")
    st.caption("Pick any product and a customer preference. The Three-Tower model finds similar products that customer is most likely to buy, "
               "and the app tells you how each one sells, what the topic's supply and demand look like, and whether to add or reduce.")

    three, two = load_models(*model_paths())
    visual, semantic, idx = load_feature_matrices()
    validation = load_customer_validation()
    report = load_sampling_report()
    pool_all = prods[prods["has_image"] & prods["article_id"].isin(idx.keys())].copy()
    if pool_all.empty:
        st.error("No products with photos and model features in this export.")
        return
    neutral = neutral_inputs(visual, semantic, validation, report.get("global_prior", {}))

    # ---------------------------------------------------------------- step 1: product
    st.markdown("#### 1. Choose a product")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        f1, f2 = st.columns([1, 1])
        topic_f = f1.selectbox("Topic filter", ["All topics"] + list(range(10)), key="st_topic_f",
                               format_func=lambda t: t if isinstance(t, str) else f"T{t} · {preference_full_name(t)}")
        q = f2.text_input("Search by name, type or colour", "", key="st_q")
        view = pool_all
        if topic_f != "All topics":
            view = view[view["topic"] == topic_f]
        if q.strip():
            ql = q.strip().lower()
            hay = (view["prod_name"].astype(str) + " " + view.get("product_type_name", "").astype(str) + " " + view.get("colour_group_name", "").astype(str)).str.lower()
            view = view[hay.str.contains(ql, regex=False)]
        view = view.sort_values("units", ascending=False, na_position="last") if has_sales(view) else view.sort_values("prod_name")
        view = view.head(400)
        if view.empty:
            st.info("No product matches. Clear the search or the topic filter.")
            return
        ids = view["article_id"].tolist()
        lab = {r["article_id"]: f"{r['prod_name'][:40]} · {r.get('product_type_name', '')} · {r.get('colour_group_name', '')}  (T{int(r['topic'])})" for _, r in view.iterrows()}
        sel_id = st.selectbox(f"Product ({len(ids)} shown, best sellers first)" if has_sales(view) else f"Product ({len(ids)} shown)", ids, format_func=lambda a: lab[a], key="st_sel")
    sel = pool_all[pool_all["article_id"] == sel_id].iloc[0]
    kt = int(sel["topic"])
    with c2:
        i1, i2 = st.columns([1, 1.2])
        i1.image(image_path(sel_id), use_container_width=True)
        pi = price_index(sel.get("avg_price"))
        i2.markdown(f"**{sel['prod_name']}**<br><span style='color:#8A8578'>{sel.get('product_type_name', '')} · {sel.get('colour_group_name', '')}</span><br>"
                    f"{chip(f'T{kt} {preference_name(kt)}', intention_color(kt))}<br>"
                    + (f"Price index <b>{pi:.0f}</b><br>" if pi == pi else "")
                    + (f"<b>{int(sel['units']):,}</b> sold · {seller_label(sel.get('units_rank_pct'))}" if sel.get("units") == sel.get("units") else ""),
                    unsafe_allow_html=True)

    # ---------------------------------------------------------------- step 2: customer
    st.markdown("#### 2. Choose the customer preference to serve")
    mode = st.radio("Customer", ["Typical customer of this product's topic", "Typical customer of another topic", "A specific customer from the demo sample"],
                    horizontal=True, key="st_mode", label_visibility="collapsed")
    if mode.startswith("Typical customer of this"):
        ck = kt
        cust = typical_customer(ck, validation)
    elif mode.startswith("Typical customer of another"):
        ck = st.selectbox("Customer's preference topic", list(range(10)), index=min(kt + 1, 9) if kt < 9 else 0, key="st_ck",
                          format_func=lambda t: f"T{t} · {preference_full_name(t)}")
        cust = typical_customer(ck, validation)
    else:
        labs = dict(zip(validation["customer_id"], validation["persona_label"]))
        cid = st.selectbox("Customer", validation["customer_id"].tolist(), format_func=lambda c: f"{labs[c]} · {c[:8]}…", key="st_cid")
        cust = one_customer(validation[validation["customer_id"] == cid].iloc[0])
        ck = int(validation.loc[validation["customer_id"] == cid, "dominant_intention"].iloc[0])
    if cust is None:
        st.warning("The demo sample has no customer of this topic.")
        return
    top_pref = np.argsort(cust["vec"])[::-1][:3]
    st.markdown(f"**{cust['label']}** · profile confidence **{cust['confidence']:.2f}** · prefers "
                + ", ".join(f"T{int(t)} {preference_name(int(t))} ({cust['vec'][t] * 100:.0f}%)" for t in top_pref))

    c1, c2, c3 = st.columns([1, 1, 1])
    n_rec = c1.slider("Products to recommend", 4, 12, 8, key="st_n")
    only_pref = c2.checkbox("Only products of the customer's preference topic", value=False, key="st_only")
    pool_n = c3.slider("Look among the most similar", 20, 120, 60, step=10, key="st_pool")

    # ---------------------------------------------------------------- scoring
    cand = pool_all[pool_all["article_id"] != sel_id]
    if only_pref:
        cand = cand[cand["topic"] == ck]
    cand = similar_ids(sel_id, cand, visual, semantic, idx, pool_n)
    scored = score_items(three, two, pd.concat([sel.to_frame().T.assign(similarity=1.0), cand], ignore_index=True), visual, semantic, idx, cust, neutral)
    me, recs = scored.iloc[0], scored.iloc[1:].copy()
    recs["rank_two"] = recs["two_p"].rank(ascending=False, method="first").astype(int)
    recs = recs.sort_values("three_p", ascending=False).head(n_rec).reset_index(drop=True)
    recs["rank_three"] = np.arange(1, len(recs) + 1)

    # ---------------------------------------------------------------- step 3: results
    thin_rule()
    st.markdown("#### 3. How well the product fits this customer")
    m = st.columns(4)
    m[0].markdown(tile("Three-Tower match", f"{me['three_p'] * 100:.1f}%", "model score for this customer"), unsafe_allow_html=True)
    m[1].markdown(tile("Two-Tower match", f"{me['two_p'] * 100:.1f}%", "same model without the preference tower"), unsafe_allow_html=True)
    m[2].markdown(tile("Preference tower adds", f"{me['uplift_pp']:+.1f}pp", "Three-Tower minus Two-Tower"), unsafe_allow_html=True)
    m[3].markdown(tile("Customer profile confidence", f"{cust['confidence']:.2f}", "how sharply the profile points to a topic"), unsafe_allow_html=True)
    st.markdown("**Share of each tower in this score**")
    _towers_bar(me, "st_me_bar")
    st.caption(f"Exact model arithmetic: each tower's input is replaced by a neutral one (the average image, the average text and customer details, the average customer's "
               f"preferences) and the drop in score is measured. Dropping Tower 3 changes the score by {me['drop_preference']:+.1f} points of probability, "
               f"Tower 1 by {me['drop_visual']:+.1f} and Tower 2 by {me['drop_text']:+.1f}. Shares are those drops as a percentage of their total. "
               "A probability of 50% means no signal: the model was trained with one non-purchase for every purchase, so read the values as match scores for ranking.")

    thin_rule()
    st.markdown(f"#### 4. Similar products this customer is most likely to buy")
    product_cards(recs, cols=4, show_action=True, extra=_rec_extra)

    show = recs.assign(Rank=recs["rank_three"], Product=recs["prod_name"], Topic=[f"T{int(t)} {preference_name(int(t))}" for t in recs["topic"]],
                       Similarity=recs["similarity"] * 100, **{"Three-Tower %": recs["three_p"] * 100, "Two-Tower %": recs["two_p"] * 100,
                                                                "Uplift (pp)": recs["uplift_pp"], "Two-Tower rank": recs["rank_two"]})[
        ["Rank", "Product", "Topic", "Similarity", "Three-Tower %", "Two-Tower %", "Uplift (pp)", "Two-Tower rank"]]
    with st.expander("All numbers"):
        st.dataframe(show, hide_index=True, use_container_width=True, column_config={
            "Similarity": st.column_config.NumberColumn(format="%.0f%%"), "Three-Tower %": st.column_config.NumberColumn(format="%.1f"),
            "Two-Tower %": st.column_config.NumberColumn(format="%.1f"), "Uplift (pp)": st.column_config.NumberColumn(format="%+.1f")})

    thin_rule()
    st.markdown("#### 5. Supply, demand and the brand's intention")
    topics = sorted({kt, ck} | set(int(t) for t in recs["topic"]))
    st_ = {t: segment_static(t) for t in topics}
    fig = go.Figure()
    nm = [f"T{t} · {preference_name(t)}" for t in topics]
    fig.add_trace(go.Bar(x=nm, y=[st_[t]["cat_share"] for t in topics], name="Share of the range (supply)", marker_color="#B8B4CF"))
    fig.add_trace(go.Bar(x=nm, y=[st_[t]["demand_share"] for t in topics], name="Share of purchases (demand)", marker_color=THREAD))
    fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=10, b=10), font=dict(family=FONT, size=13),
                      legend=dict(orientation="h", y=1.12, x=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title="%", gridcolor="#ECE9E1"); fig.update_xaxes(tickangle=-20)
    st.plotly_chart(fig, use_container_width=True, key="st_sd")
    for t in topics:
        e, l, c = status_badge(st_[t]["gap"])
        st.markdown(f"- **T{t} · {preference_name(t)}**: {e} {l}, purchases {st_[t]['demand_share']:.1f}% vs range {st_[t]['cat_share']:.1f}% ({st_[t]['gap']:+.1f}pp)")

    acts = []
    for _, r in recs.iterrows():
        lab_, why, col = action(segment_static(int(r["topic"]))["gap"], r.get("units_rank_pct"))
        acts.append((r, lab_, why, col))
    sel_lab, sel_why, sel_col = action(segment_static(kt)["gap"], sel.get("units_rank_pct"))
    st.markdown(f"**The product you chose** — {action_chip(sel_lab, sel_col)} &nbsp; {sel_why}", unsafe_allow_html=True)
    if acts:
        st.markdown("**The recommended products**")
        for lab_ in ["Add more like this", "Add, but pick stronger sellers", "Maintain & promote", "Maintain", "Keep, do not add", "Review", "Reduce / rationalise"]:
            grp = [a for a in acts if a[1] == lab_]
            if grp:
                st.markdown(f"{action_chip(lab_, grp[0][3])} &nbsp; " + ", ".join(f"{a[0]['prod_name'][:30]} (T{int(a[0]['topic'])})" for a in grp), unsafe_allow_html=True)
        st.caption("Rule: the topic's supply-demand gap (5pp = monitor, as in Section 9) combined with the product's seller rank inside its topic "
                   "(top 30% sells well, bottom 30% sells poorly). Sales are real units over the whole dataset.")
