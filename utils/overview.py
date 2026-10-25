"""
overview.py
================================================================================
Page 1 "Overview": the whole analysis on one screen, straight from the Kaggle
outputs. Every table and chart names the file it was read from.

Tabs: Key figures · Data & customers · Preference topics · Training data · Model results
================================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import kaggle_data as kd
from utils.preference_labels import preference_name, product_line
from utils.theme import intention_color, thin_rule, THREAD, SIGNAL, MUTED
from utils.thesis_data import segment_static, model_improvement

FONT = "IBM Plex Sans, sans-serif"


def _style(fig, height, title=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=44 if title else 10, b=10),
                      title=dict(text=title, font=dict(size=15), x=0) if title else None,
                      font=dict(family=FONT, size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _kpi(col, label, value, note=None):
    with col:
        st.metric(label, value)
        if note:
            st.caption(note)


def _src(*files):
    st.caption("Source: " + ", ".join(f"`{f}`" for f in files))


def _topic_label(k):
    return f"T{int(k)} · {preference_name(int(k))}"


# --------------------------------------------------------------------------- KPI strip
def render_kpi_strip():
    ov, ev, sp, lm = kd.overview(), kd.evaluation(), kd.split_stats(), kd.lda_meta()
    t3, t2 = ev.get("three_tower", {}), ev.get("two_tower", {})
    c = st.columns(6)
    _kpi(c[0], "Customers", kd.n(ov.get("n_customers")), f"{kd.n(kd.user_stats().get('active_customers'))} have purchased")
    _kpi(c[1], "Products", kd.n(ov.get("n_articles")))
    _kpi(c[2], "Transactions", kd.n(ov.get("n_transactions")), f"{ov.get('date_start', '')} to {ov.get('date_end', '')}")
    _kpi(c[3], "Preference topics", kd.n(lm.get("optimal_K")), "found from product attributes")
    _kpi(c[4], "AUC, Three-Tower", kd.f(t3.get("auc")), f"vs {kd.f(t2.get('auc'))} Two-Tower ({kd.rel_gain(t3.get('auc'), t2.get('auc')):+.2f}%)")
    _kpi(c[5], "Test rows", kd.n(sp.get("n_test_total")), f"cut-off {sp.get('cutoff_date', '')}")


# --------------------------------------------------------------------------- tab: key figures
def tab_key_figures():
    st.markdown("The important numbers of the whole analysis, each with the Kaggle file it was read from.")
    kf = kd.key_figures()
    for section in kf["Section"].unique():
        st.markdown(f"**{section}**")
        st.dataframe(kf[kf["Section"] == section][["Key figure", "Value", "Source file"]], hide_index=True,
                     use_container_width=True, height=35 * (int((kf["Section"] == section).sum()) + 1) + 3)


# --------------------------------------------------------------------------- tab: data & customers
def tab_data():
    ov, us, fs = kd.overview(), kd.user_stats(), kd.feature_summary()
    price = ov.get("price", {})
    st.subheader("The H&M data")
    c = st.columns(4)
    _kpi(c[0], "Products", kd.n(ov.get("n_articles")), f"{kd.n(ov.get('n_product_types'))} product types" if ov.get("n_product_types") else None)
    _kpi(c[1], "Customers", kd.n(ov.get("n_customers")))
    _kpi(c[2], "Transactions", kd.n(ov.get("n_transactions")))
    _kpi(c[3], "Days with sales", kd.n(ov.get("n_days_with_sales")), f"{ov.get('date_start', '')} to {ov.get('date_end', '')}")
    _src("dataset_overview.json")

    st.markdown("**Price per item** (normalised by the dataset provider, not currency)")
    pc = st.columns(5)
    for col, (lab, key) in zip(pc, (("Min", "min"), ("25%", "q25"), ("Median", "median"), ("75%", "q75"), ("Max", "max"))):
        _kpi(col, lab, kd.f(price.get(key)))
    _src("dataset_overview.json")

    thin_rule()
    st.subheader("Customers")
    c = st.columns(4)
    _kpi(c[0], "Active customers", kd.n(us.get("active_customers")), "at least one purchase")
    _kpi(c[1], "New customers", kd.n(us.get("cold_start_customers")), f"{float(us.get('cold_start_rate', 0)) * 100:.2f}% with no purchase yet")
    _kpi(c[2], "Purchases per customer", kd.f(us.get("mean_purchases"), 1), f"median {kd.f(us.get('median_purchases'), 1)}, max {kd.n(us.get('max_purchases'))}")
    _kpi(c[3], "Mean profile confidence", kd.f(us.get("mean_confidence_active")), "active customers")
    st.markdown(
        "A customer's **preference profile** is built from what they buy. Recent purchases count more "
        f"(decay λ = {us.get('recency_lambda', 'n/a')}), and each purchase is weighted by its price tier "
        f"(cut-offs {kd.f(us.get('price_quartiles', {}).get('q25'))} / {kd.f(us.get('price_quartiles', {}).get('q50'))} / "
        f"{kd.f(us.get('price_quartiles', {}).get('q75'))}), so frequent and higher-spend purchases shape the profile most. "
        f"The profile is smoothed towards the average customer (weights {us.get('bayesian_alpha', 'n/a')} / {us.get('bayesian_beta', 'n/a')}), "
        "which is what lets the model say something about new customers."
    )
    pw = us.get("price_weights")
    if isinstance(pw, dict):
        st.caption("Price tier weights: " + ", ".join(f"{k} {v}" for k, v in pw.items()))
    _src("user_profiling_stats.json")

    thin_rule()
    st.subheader("What the topic model reads from every product")
    if not fs.empty and "n_features" in fs:
        show = fs.copy()
        show.columns = [str(c).replace("_", " ").capitalize() for c in show.columns]
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.caption(f"Total: {kd.n(fs['n_features'].sum())} features per product: colour, pattern, material and description, "
                   "category and price level. The neural network also reads the product photo and the product text "
                   f"({kd.n(kd.split_stats().get('feature_dims', {}).get('tower1_visual'))} image features, "
                   f"{kd.n(kd.split_stats().get('feature_dims', {}).get('tower2_semantic'))} text features).")
        _src("lda_feature_summary.csv", "split_statistics.json")


# --------------------------------------------------------------------------- tab: preference topics
def _topic_table() -> pd.DataFrame:
    summ, price = kd.topic_summary(), kd.price_stats()
    rows = []
    for k in range(10):
        s = segment_static(k)
        a = summ[summ["topic"] == k]
        pr = price[price["topic"] == k] if not price.empty else price
        rows.append({
            "Topic": _topic_label(k), "Product line": product_line(k),
            "Products": int(a["article_count"].iloc[0]) if len(a) else 0,
            "% of catalogue": s["cat_share"], "% of purchases": s["demand_share"], "Gap (pp)": s["gap"],
            "Mean price": float(a["mean_price"].iloc[0]) if len(a) else np.nan,
            "Price tier": str(pr["price_tier"].iloc[0]) if len(pr) and "price_tier" in pr else "",
            "Topic concentration": s["confidence"],
            "Customers": s["users"], "% of customers": s["user_share"],
            "AUC gain (%)": model_improvement(k),
        })
    return pd.DataFrame(rows)


def tab_topics():
    el, lm = kd.elbow(), kd.lda_meta()
    K = int(float(lm.get("optimal_K", 10)))
    st.subheader("How many topics?")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        if not el.empty:
            fig = go.Figure(go.Scatter(x=el["K"], y=el["perplexity"], mode="lines+markers", line=dict(color=THREAD, width=3),
                                       marker=dict(size=8), name="Perplexity",
                                       hovertemplate="K = %{x}<br>perplexity %{y:.2f}<extra></extra>"))
            hit = el[el["K"] == K]
            if len(hit):
                fig.add_trace(go.Scatter(x=hit["K"], y=hit["perplexity"], mode="markers", marker=dict(size=16, color=SIGNAL, symbol="star"),
                                         name=f"K = {K}", hovertemplate=f"K = {K}<extra></extra>"))
            fig.update_xaxes(title="Number of topics (K)", dtick=2, gridcolor="#ECE9E1")
            fig.update_yaxes(title="Perplexity (lower is better)", gridcolor="#ECE9E1")
            fig.update_layout(showlegend=False)
            _style(fig, 360)
            st.plotly_chart(fig, use_container_width=True, key="ov_elbow")
            _src("elbow_results.csv")
    with c2:
        v = kd.validation()
        if not v.empty:
            show = v[["test", "statistic", "p_value"]].rename(columns={"test": "Test", "statistic": "Statistic", "p_value": "p-value"})
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.caption("These tests check that the 10 topics are genuinely different from each other (price) and that "
                       "demand is not proportional to the catalogue.")
            _src("topic_validation.csv")

    thin_rule()
    st.subheader("The 10 preference topics")
    tbl = _topic_table()
    st.dataframe(tbl, hide_index=True, use_container_width=True, column_config={
        "Products": st.column_config.NumberColumn(format="%d"), "Customers": st.column_config.NumberColumn(format="%d"),
        "% of catalogue": st.column_config.NumberColumn(format="%.2f%%"), "% of purchases": st.column_config.NumberColumn(format="%.2f%%"),
        "% of customers": st.column_config.NumberColumn(format="%.2f%%"), "Gap (pp)": st.column_config.NumberColumn(format="%+.2f"),
        "Mean price": st.column_config.NumberColumn(format="%.4f"), "Topic concentration": st.column_config.NumberColumn(format="%.4f"),
        "AUC gain (%)": st.column_config.NumberColumn(format="%+.2f"),
    })
    st.caption("Topic concentration = mean probability of the topic among its own products (higher = a more clearly defined preference). "
               "Gap = share of purchases minus share of the catalogue. AUC gain = Three-Tower over Two-Tower for the topic's products.")
    _src("topic_summary.csv", "topic_gap_analysis.csv", "topic_price_stats.csv", "user_dominant_topic_dist.csv / population", "per_topic_metrics.csv")

    thin_rule()
    st.subheader("Supply and demand")
    g = kd.gap()
    if not g.empty:
        order = g.sort_values("gap_pp", ascending=True)
        fig = go.Figure(go.Bar(
            x=order["gap_pp"], y=[_topic_label(k) for k in order["topic"]], orientation="h",
            marker_color=["#E4572E" if v > 0 else "#2F6DA3" for v in order["gap_pp"]],
            text=[f"{v:+.1f}pp" for v in order["gap_pp"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>gap %{x:+.2f}pp<extra></extra>"))
        fig.add_vline(x=10, line_dash="dash", line_color="#E4572E", opacity=0.6)
        fig.add_vline(x=-10, line_dash="dash", line_color="#2F6DA3", opacity=0.6)
        fig.add_vline(x=0, line_color=MUTED)
        lim = max(15, float(np.abs(g["gap_pp"]).max()) + 4)
        fig.update_xaxes(range=[-lim, lim], title="Share of purchases minus share of catalogue (percentage points)", gridcolor="#ECE9E1")
        _style(fig, 430, "Red: demand is higher than the range. Blue: the range is larger than demand. Dashed lines: ±10pp.")
        st.plotly_chart(fig, use_container_width=True, key="ov_gap")
        big = g[g["gap_pp"].abs() >= 10]
        if len(big):
            st.markdown("- " + "\n- ".join(
                f"**{_topic_label(r.topic)}**: {r.gap_pp:+.2f}pp — {'demand exceeds the range' if r.gap_pp > 0 else 'the range exceeds demand'}"
                for r in big.itertuples()))
        _src("topic_gap_analysis.csv")

    with st.expander("What defines each topic: its 20 strongest product features"):
        tf = kd.top_features()
        if tf.empty:
            st.caption("`topic_top_features.csv` is not in this export.")
        else:
            for k in range(10):
                sub = tf[tf["topic"] == k].sort_values("rank")
                st.markdown(f"**{_topic_label(k)}**  \n" + ", ".join(f"`{x}`" for x in sub["feature"].tolist()))
            _src("topic_top_features.csv")


# --------------------------------------------------------------------------- tab: training data
def tab_training():
    sp, ss = kd.split_stats(), kd.sampling_stats()
    st.subheader("How the training data was built")
    st.markdown(
        f"**1. Sample customers.** {kd.pct(float(sp.get('sample_rate', 0)) * 100, 0)} of active customers are drawn from each "
        f"preference topic, giving **{kd.n(sp.get('n_sampled_users'))}** customers out of {kd.n(sp.get('n_active_population'))}.  \n"
        f"**2. Split by time.** Purchases up to **{sp.get('cutoff_date', 'n/a')}** train the model; later purchases test it. "
        f"{kd.n(sp.get('n_train_raw'))} train and {kd.n(sp.get('n_test_raw'))} test transactions, "
        f"after removing {kd.n(sp.get('n_overlap_removed'))} repeated (customer, product) pairs from the test side.  \n"
        f"**3. Add non-purchases.** For every purchase, {kd.n(sp.get('neg_ratio'))} product the customer did not buy is drawn at random."
    )
    c = st.columns(4)
    _kpi(c[0], "Train rows", kd.n(sp.get("n_train_total")), f"{kd.n(sp.get('n_train_positive'))} purchases + {kd.n(sp.get('n_train_negative'))} non-purchases")
    _kpi(c[1], "Test rows", kd.n(sp.get("n_test_total")), f"{kd.n(sp.get('n_test_positive'))} purchases + {kd.n(sp.get('n_test_negative'))} non-purchases")
    _kpi(c[2], "Test customers", kd.n(sp.get("n_test_users")), f"{kd.n(sp.get('n_warm_test_users'))} also in train, {kd.n(sp.get('n_cold_test_users'))} new")
    _kpi(c[3], "Products", f"{kd.n(sp.get('n_train_items'))} in train", f"{kd.n(sp.get('n_test_items'))} in test")
    _src("split_statistics.json")

    thin_rule()
    st.subheader("The sample keeps the topic mix")
    if not ss.empty:
        x = [_topic_label(k) for k in ss["topic"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=ss["pop_share"] * 100, name="All active customers", marker_color="#B8B4CF"))
        fig.add_trace(go.Bar(x=x, y=ss["sample_share"] * 100, name="Sample", marker_color=THREAD))
        fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.1, x=0))
        fig.update_yaxes(title="% of customers", gridcolor="#ECE9E1"); fig.update_xaxes(tickangle=-35)
        _style(fig, 400)
        st.plotly_chart(fig, use_container_width=True, key="ov_sample")
        dev = (ss["sample_share"] - ss["pop_share"]).abs().max() * 100
        st.caption(f"Largest difference between the sample and all customers: {dev:.4f} percentage points "
                   f"(reported by Section 11: {kd.f(sp.get('max_distribution_deviation_pp'), 4)}).")
        show = ss.assign(Topic=[_topic_label(k) for k in ss["topic"]])[["Topic", "n_population", "n_sampled"]].rename(
            columns={"n_population": "Active customers", "n_sampled": "Sampled"})
        st.dataframe(show, hide_index=True, use_container_width=True)
        _src("sampling_statistics.csv", "split_statistics.json")


# --------------------------------------------------------------------------- tab: model results
def _metric_table() -> pd.DataFrame:
    ev = kd.evaluation()
    t3, t2, rk = ev.get("three_tower", {}), ev.get("two_tower", {}), ev.get("ranking_metrics", {})
    rows = [("AUC", t2.get("auc"), t3.get("auc")), ("Accuracy", t2.get("accuracy"), t3.get("accuracy")),
            ("Precision", t2.get("precision"), t3.get("precision")), ("Recall", t2.get("recall"), t3.get("recall")),
            ("F1", t2.get("f1"), t3.get("f1")), ("NDCG@10", rk.get("two_tower_ndcg"), rk.get("three_tower_ndcg")),
            ("MAP", rk.get("two_tower_map"), rk.get("three_tower_map"))]
    return pd.DataFrame([{"Metric": m, "Two-Tower": a, "Three-Tower": b, "Difference": (b - a) if a is not None and b is not None else np.nan,
                          "Change (%)": kd.rel_gain(b, a)} for m, a, b in rows])


def tab_model():
    ev, tm, wm = kd.evaluation(), kd.three_metrics(), kd.two_metrics()
    st.subheader("Two models, one difference")
    st.markdown(
        "The **Two-Tower** model reads how a product looks (image) and what it means (text, plus basic customer data). "
        "The **Three-Tower** model adds one more tower: how well the product matches the customer's **preference profile**. "
        "Everything else is identical, so any difference in results comes from that third tower."
    )
    c = st.columns(4)
    d_par = (tm.get("n_parameters") or 0) - (wm.get("n_parameters") or 0)
    _kpi(c[0], "Parameters, Three-Tower", kd.n(tm.get("n_parameters")), f"Two-Tower {kd.n(wm.get('n_parameters'))} (+{kd.n(d_par)}, {d_par / max(wm.get('n_parameters') or 1, 1) * 100:.2f}%)")
    _kpi(c[1], "Best epoch", f"{kd.n(tm.get('best_epoch'))} of {kd.n(tm.get('n_epochs'))}", f"Two-Tower: epoch {kd.n(wm.get('best_epoch'))}")
    _kpi(c[2], "Training time", f"{kd.f(tm.get('train_time_min'), 1)} min", f"Two-Tower {kd.f(wm.get('train_time_min'), 1)} min ({tm.get('device', '')})")
    _kpi(c[3], "Test rows scored", kd.n(ev.get("test_samples")))
    _src("three_tower_metrics.json", "two_tower_metrics.json", "evaluation_metrics.json")

    thin_rule()
    st.subheader("Results on the test set")
    mt = _metric_table()
    st.dataframe(mt, hide_index=True, use_container_width=True, column_config={
        "Two-Tower": st.column_config.NumberColumn(format="%.4f"), "Three-Tower": st.column_config.NumberColumn(format="%.4f"),
        "Difference": st.column_config.NumberColumn(format="%+.4f"), "Change (%)": st.column_config.NumberColumn(format="%+.2f%%")})
    st.caption("AUC measures how well the model ranks purchases above non-purchases (0.5 = random, 1 = perfect). Precision, recall, accuracy and F1 use a 0.5 threshold.")
    _src("evaluation_metrics.json")

    stt = ev.get("statistical_tests", {})
    if stt:
        tt, wl = stt.get("paired_ttest", {}), stt.get("wilcoxon", {})
        st.markdown(f"Across all {kd.n(ev.get('test_samples'))} test rows the Three-Tower scores are higher: paired t-test "
                    f"t = {kd.f(tt.get('t_statistic'), 2)} (p {kd.pval(tt.get('p_value'))}), Wilcoxon test W = {kd.n(wl.get('w_statistic'))} (p {kd.pval(wl.get('p_value'))}).")

    t3auc, t2auc = tm.get("test_auc"), wm.get("test_auc")
    ev3, ev2 = ev.get("three_tower", {}).get("auc"), ev.get("two_tower", {}).get("auc")
    if None not in (t3auc, t2auc, ev3, ev2) and (abs(t3auc - ev3) > 5e-5 or abs(t2auc - ev2) > 5e-5):
        st.caption(f"The training notebooks reported test AUC {kd.f(t3auc)} (Three-Tower) and {kd.f(t2auc)} (Two-Tower); the final evaluation above "
                   f"re-scored the full test set and gives {kd.f(ev3)} and {kd.f(ev2)}. The final evaluation is the one used here.")

    thin_rule()
    st.subheader("Learning curves")
    l3, l2 = kd.three_log(), kd.two_log()
    if not l3.empty and not l2.empty:
        fig = go.Figure()
        for name, df, col, best in (("Three-Tower", l3, THREAD, tm.get("best_epoch")), ("Two-Tower", l2, SIGNAL, wm.get("best_epoch"))):
            fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_auc"], mode="lines+markers", name=name, line=dict(color=col, width=3)))
            hit = df[df["epoch"] == best]
            if len(hit):
                fig.add_trace(go.Scatter(x=hit["epoch"], y=hit["val_auc"], mode="markers", marker=dict(size=15, color=col, symbol="star"),
                                         showlegend=False, hovertemplate=f"{name} best epoch %{{x}}<extra></extra>"))
        fig.update_xaxes(title="Epoch", dtick=1, gridcolor="#ECE9E1"); fig.update_yaxes(title="Validation AUC", gridcolor="#ECE9E1")
        fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0))
        _style(fig, 400, "Stars mark the epoch whose weights were kept")
        st.plotly_chart(fig, use_container_width=True, key="ov_curves")
        _src("three_tower_training_log.csv", "two_tower_training_log.csv")

    thin_rule()
    st.subheader("Where the third tower helps most")
    pt = kd.per_topic()
    if not pt.empty:
        x = [_topic_label(k) for k in pt["intention"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=pt["two_tower_auc"], name="Two-Tower", marker_color="#B8B4CF"))
        fig.add_trace(go.Bar(x=x, y=pt["three_tower_auc"], name="Three-Tower", marker_color=THREAD))
        fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.1, x=0))
        lo = max(0.5, float(min(pt["two_tower_auc"].min(), pt["three_tower_auc"].min())) - 0.05)
        fig.update_yaxes(title="AUC", range=[lo, 1.0], gridcolor="#ECE9E1"); fig.update_xaxes(tickangle=-35)
        _style(fig, 420)
        st.plotly_chart(fig, use_container_width=True, key="ov_pertopic")
        show = pt.assign(Topic=x, **{"Gain (AUC points)": pt["auc_improvement"], "Gain (%)": [kd.rel_gain(b, a) for a, b in zip(pt["two_tower_auc"], pt["three_tower_auc"])]})[
            ["Topic", "n_samples", "two_tower_auc", "three_tower_auc", "Gain (AUC points)", "Gain (%)"]].rename(
            columns={"n_samples": "Test rows", "two_tower_auc": "Two-Tower AUC", "three_tower_auc": "Three-Tower AUC"}).sort_values("Gain (%)", ascending=False)
        st.dataframe(show, hide_index=True, use_container_width=True, column_config={
            "Two-Tower AUC": st.column_config.NumberColumn(format="%.4f"), "Three-Tower AUC": st.column_config.NumberColumn(format="%.4f"),
            "Gain (AUC points)": st.column_config.NumberColumn(format="%+.4f"), "Gain (%)": st.column_config.NumberColumn(format="%+.2f%%"), "Test rows": st.column_config.NumberColumn(format="%d")})
        _src("per_topic_metrics.csv")

    _extras_block()


def _extras_block():
    ex, roc, cm, cs = kd.extras(), kd.roc_points(), kd.confusion(), kd.cold_start()
    thin_rule()
    st.subheader("More detail on the comparison")
    if not (ex or not roc.empty or cm or not cs.empty):
        st.info("ROC curve, confusion-rate difference, per-customer tests (Cohen's d), McNemar test and new-customer results are not in this export. "
                "Run PART 4 of `kaggle/HM_KAGGLE_EXPORT_ALL.ipynb` to add them.")
        return
    c1, c2 = st.columns(2)
    with c1:
        if not roc.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr_three_tower"], mode="lines", name=f"Three-Tower (AUC {kd.f(roc['auc_three_tower'].iloc[0])})", line=dict(color=THREAD, width=3)))
            fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr_two_tower"], mode="lines", name=f"Two-Tower (AUC {kd.f(roc['auc_two_tower'].iloc[0])})", line=dict(color=SIGNAL, width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=MUTED, dash="dash")))
            fig.update_xaxes(title="False positive rate", gridcolor="#ECE9E1"); fig.update_yaxes(title="True positive rate", gridcolor="#ECE9E1")
            fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0))
            _style(fig, 400, "ROC curve")
            st.plotly_chart(fig, use_container_width=True, key="ov_roc")
            _src("roc_curve_points.csv")
    with c2:
        if cm:
            d = np.array(cm.get("rate_difference_three_minus_two", [[0, 0], [0, 0]]))
            fig = go.Figure(go.Heatmap(z=d, x=["Predicted: no", "Predicted: yes"], y=["Actual: not bought", "Actual: bought"],
                                       text=[[f"{v:+.3f}" for v in row] for row in d], texttemplate="%{text}", colorscale="RdBu", zmid=0, showscale=False))
            fig.update_yaxes(autorange="reversed")
            _style(fig, 400, "Three-Tower minus Two-Tower, share of each actual group")
            st.plotly_chart(fig, use_container_width=True, key="ov_cm")
            st.caption("Bottom-right: more real purchases found. Top-right: change in false alarms.")
            _src("confusion_matrices.json")
    if ex:
        ul, mc = ex.get("user_level_paired_tests", {}), ex.get("mcnemar", {})
        st.markdown(
            f"- **Per customer.** Across {kd.n(ul.get('n_users_with_both_classes'))} test customers, mean AUC is {kd.f(ul.get('mean_auc_three_tower'))} "
            f"(Three-Tower) vs {kd.f(ul.get('mean_auc_two_tower'))} (Two-Tower). Paired t = {kd.f(ul.get('paired_t'), 2)} (p {kd.pval(ul.get('p_value_t'))}); "
            f"effect size Cohen's d = {kd.f(ul.get('cohens_d_paired'), 3)}, 95% CI {ul.get('cohens_d_ci95')} ({ul.get('cohens_d_label', '')}).\n"
            f"- **Where they disagree.** The Three-Tower is right in {kd.pct(mc.get('three_tower_share_of_disagreements_pct'), 1)} of the cases where exactly one model is right "
            f"(McNemar χ² = {kd.f(mc.get('chi2_continuity_corrected'), 1)}, p {kd.pval(mc.get('p_value'))})."
        )
        _src("evaluation_extras.json")
    if not cs.empty:
        st.markdown("**New and returning customers, new and known products**")
        show = cs.rename(columns={"user_type": "Customer", "item_type": "Product", "n_rows": "Rows", "n_users": "Customers", "n_items": "Products",
                                  "n_positive": "Purchases", "auc_two_tower": "Two-Tower AUC", "auc_three_tower": "Three-Tower AUC", "auc_delta": "Difference"})
        st.dataframe(show, hide_index=True, use_container_width=True, column_config={
            "Two-Tower AUC": st.column_config.NumberColumn(format="%.4f"), "Three-Tower AUC": st.column_config.NumberColumn(format="%.4f"), "Difference": st.column_config.NumberColumn(format="%+.4f")})
        if ex:
            st.caption(f"Known product = appears in the training rows. {kd.n(ex.get('test_items_seen_in_training'))} of {kd.n(ex.get('test_items_total'))} test products are known, "
                       "so results for unknown products rest on very few rows.")
        _src("cold_start_breakdown.csv")


def render_overview_tabs():
    t = st.tabs(["Key figures", "Data & customers", "Preference topics", "Training data", "Model results"])
    with t[0]: tab_key_figures()
    with t[1]: tab_data()
    with t[2]: tab_topics()
    with t[3]: tab_training()
    with t[4]: tab_model()
