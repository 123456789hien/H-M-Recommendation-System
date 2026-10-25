"""
kaggle_data.py
================================================================================
The single source of truth for every number the app shows.

Everything is read from the files the Kaggle notebooks (Sections 1-14) wrote,
collected by PART 3 of kaggle/HM_KAGGLE_EXPORT_ALL.ipynb into  data/kaggle_outputs/  (and
data/population/  for the population analysis). Nothing here is typed in by
hand: if a file is missing the app says so instead of showing a stand-in.
================================================================================
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import DATA_DIR

KO_SUBDIR = "kaggle_outputs"

# Files the app cannot run without (name, Kaggle section)
ESSENTIAL = [
    ("topic_summary.csv", "Section 8"), ("topic_gap_analysis.csv", "Section 9"),
    ("topic_validation.csv", "Section 9"), ("user_profiling_stats.json", "Section 10"),
    ("user_dominant_topic_dist.csv", "Section 10"), ("split_statistics.json", "Section 11"),
    ("three_tower_metrics.json", "Section 12"), ("two_tower_metrics.json", "Section 13"),
    ("evaluation_metrics.json", "Section 14"), ("per_topic_metrics.csv", "Section 14"),
    ("dataset_overview.json", "Part 3B"),
]


def ko_path(name: str) -> str:
    return os.path.join(DATA_DIR, KO_SUBDIR, name)


def missing_essential() -> list[tuple[str, str]]:
    return [(n, s) for n, s in ESSENTIAL if not os.path.exists(ko_path(n))]


def require_kaggle_outputs():
    """Call once per page, right after download_data(). Stops the page with a clear
    message when the export does not contain the Kaggle outputs yet."""
    miss = missing_essential()
    if miss:
        st.error(
            "The Kaggle output files this app reads are not in the data export yet:\n\n"
            + "\n".join(f"- `{n}` ({s})" for n, s in miss)
            + "\n\nRun **PART 3 and PART 5** of `kaggle/HM_KAGGLE_EXPORT_ALL.ipynb`, upload the new "
              "`data_export.zip` to Google Drive and reboot the app."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def read_csv(name: str, **kw) -> pd.DataFrame:
    p = ko_path(name)
    return pd.read_csv(p, **kw) if os.path.exists(p) else pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_json(name: str) -> dict:
    p = ko_path(name)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def has(name: str) -> bool:
    return os.path.exists(ko_path(name))


# ---- thin, named accessors (one per Kaggle file) ----
def overview() -> dict: return read_json("dataset_overview.json")
def feature_summary() -> pd.DataFrame: return read_csv("lda_feature_summary.csv")
def elbow() -> pd.DataFrame: return read_csv("elbow_results.csv")
def topic_summary() -> pd.DataFrame: return read_csv("topic_summary.csv").sort_values("topic").reset_index(drop=True)
def top_features() -> pd.DataFrame: return read_csv("topic_top_features.csv")
def validation() -> pd.DataFrame: return read_csv("topic_validation.csv")
def price_stats() -> pd.DataFrame: return read_csv("topic_price_stats.csv")
def gap() -> pd.DataFrame: return read_csv("topic_gap_analysis.csv").sort_values("topic").reset_index(drop=True)
def key_attributes() -> pd.DataFrame: return read_csv("topic_key_attributes.csv")
def correlation() -> pd.DataFrame: return read_csv("topic_correlation_matrix.csv", index_col=0)
def user_stats() -> dict: return read_json("user_profiling_stats.json")
def user_dist() -> pd.DataFrame: return read_csv("user_dominant_topic_dist.csv").sort_values("topic").reset_index(drop=True)
def sampling_stats() -> pd.DataFrame: return read_csv("sampling_statistics.csv").sort_values("topic").reset_index(drop=True)
def split_stats() -> dict: return read_json("split_statistics.json")
def three_metrics() -> dict: return read_json("three_tower_metrics.json")
def two_metrics() -> dict: return read_json("two_tower_metrics.json")
def three_log() -> pd.DataFrame: return read_csv("three_tower_training_log.csv")
def two_log() -> pd.DataFrame: return read_csv("two_tower_training_log.csv")
def evaluation() -> dict: return read_json("evaluation_metrics.json")
def per_topic() -> pd.DataFrame: return read_csv("per_topic_metrics.csv").sort_values("intention").reset_index(drop=True)
def extras() -> dict: return read_json("evaluation_extras.json")
def roc_points() -> pd.DataFrame: return read_csv("roc_curve_points.csv")
def confusion() -> dict: return read_json("confusion_matrices.json")
def cold_start() -> pd.DataFrame: return read_csv("cold_start_breakdown.csv")


def lda_meta() -> dict:
    """lda_model_metadata.csv was written with Series.to_csv: two columns (key, value)."""
    df = read_csv("lda_model_metadata.csv", header=None)
    if df.empty:
        return {}
    df = df.iloc[1:] if str(df.iloc[0, 0]).strip() in ("", "nan", "Unnamed: 0") else df
    return {str(k): v for k, v in zip(df.iloc[:, 0], df.iloc[:, 1])}


def validation_row(test_prefix: str) -> dict:
    v = validation()
    m = v[v["test"].astype(str).str.startswith(test_prefix)] if not v.empty else v
    return m.iloc[0].to_dict() if len(m) else {}


def population_dir() -> str:
    """Population tables ride along in data_export.zip; assets/population is the fallback."""
    inside = os.path.join(DATA_DIR, "population")
    if os.path.isdir(inside):
        return inside
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "population")


# ---- number formatting helpers ----
def n(v) -> str:
    try:
        return f"{int(round(float(v))):,}"
    except (TypeError, ValueError):
        return "n/a"


def f(v, d=4) -> str:
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return "n/a"


def pct(v, d=2) -> str:
    try:
        return f"{float(v):.{d}f}%"
    except (TypeError, ValueError):
        return "n/a"


def pval(p) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def rel_gain(new, old) -> float:
    try:
        return (float(new) - float(old)) / float(old) * 100
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


# ---- the key figures table (page 1): value + the file it came from ----
def key_figures() -> pd.DataFrame:
    rows: list[tuple[str, str, str, str]] = []

    def add(section, name, value, source):
        rows.append((section, name, value, source))

    ov, us, sp = overview(), user_stats(), split_stats()
    lm, ev, tm, wm = lda_meta(), evaluation(), three_metrics(), two_metrics()
    fs = feature_summary()
    price = ov.get("price", {})

    add("Data", "Products (articles)", n(ov.get("n_articles")), "dataset_overview.json")
    add("Data", "Customers", n(ov.get("n_customers")), "dataset_overview.json")
    add("Data", "Transactions", n(ov.get("n_transactions")), "dataset_overview.json")
    add("Data", "Period", f"{ov.get('date_start', 'n/a')} to {ov.get('date_end', 'n/a')}", "dataset_overview.json")
    if ov.get("n_images_available"):
        add("Data", "Product photos available", n(ov["n_images_available"]), "dataset_overview.json")
    add("Data", "Median / mean price per item (normalised)", f"{f(price.get('median'))} / {f(price.get('mean'))}", "dataset_overview.json")

    add("Customers", "Active customers (at least one purchase)", n(us.get("active_customers")), "user_profiling_stats.json")
    add("Customers", "New customers without purchases", f"{n(us.get('cold_start_customers'))} ({pct(float(us.get('cold_start_rate', np.nan)) * 100)})", "user_profiling_stats.json")
    add("Customers", "Purchases per customer: mean / median / max",
        f"{f(us.get('mean_purchases'), 1)} / {f(us.get('median_purchases'), 1)} / {n(us.get('max_purchases'))}", "user_profiling_stats.json")
    pq = us.get("price_quartiles", {})
    add("Customers", "Price tier cut-offs (Q25 / Q50 / Q75)", f"{f(pq.get('q25'))} / {f(pq.get('q50'))} / {f(pq.get('q75'))}", "user_profiling_stats.json")

    if not fs.empty and "n_features" in fs:
        add("Features", "Topic-model input features", n(fs["n_features"].sum()), "lda_feature_summary.csv")
    fd = sp.get("feature_dims", {})
    if fd:
        add("Features", "Image features / text features per product", f"{n(fd.get('tower1_visual'))} / {n(fd.get('tower2_semantic'))}", "split_statistics.json")

    add("Preference topics", "Number of topics (K)", n(lm.get("optimal_K")), "lda_model_metadata.csv")
    add("Preference topics", "Topic model perplexity", f(lm.get("perplexity_section8"), 2), "lda_model_metadata.csv")
    for label, key in (("ANOVA F (price differs between topics)", "One-way ANOVA"), ("Kruskal-Wallis H", "Kruskal-Wallis"),
                       ("Effect size η²", "Eta-squared"), ("χ² supply-demand gap", "Chi-squared")):
        r = validation_row(key)
        if r:
            add("Preference topics", label, f(r.get("statistic"), 3 if key == "Eta-squared" else 2), "topic_validation.csv")

    add("Training data", "Customers sampled (stratified by topic)", f"{n(sp.get('n_sampled_users'))} ({pct(float(sp.get('sample_rate', np.nan)) * 100, 0)})", "split_statistics.json")
    add("Training data", "Chronological cut-off", str(sp.get("cutoff_date", "n/a")), "split_statistics.json")
    add("Training data", "Train rows / test rows (with 1:1 non-purchases)", f"{n(sp.get('n_train_total'))} / {n(sp.get('n_test_total'))}", "split_statistics.json")
    add("Training data", "Repeated pairs removed from test", n(sp.get("n_overlap_removed")), "split_statistics.json")

    add("Model", "Parameters: Three-Tower / Two-Tower", f"{n(tm.get('n_parameters'))} / {n(wm.get('n_parameters'))}", "three_tower_metrics.json, two_tower_metrics.json")
    add("Model", "Best epoch: Three-Tower / Two-Tower", f"{n(tm.get('best_epoch'))} / {n(wm.get('best_epoch'))}", "three_tower_metrics.json, two_tower_metrics.json")
    t3, t2 = ev.get("three_tower", {}), ev.get("two_tower", {})
    if t3 and t2:
        add("Model", "AUC: Three-Tower vs Two-Tower (test set)", f"{f(t3.get('auc'))} vs {f(t2.get('auc'))}  ({rel_gain(t3.get('auc'), t2.get('auc')):+.2f}%)", "evaluation_metrics.json")
        add("Model", "Recall: Three-Tower vs Two-Tower", f"{f(t3.get('recall'))} vs {f(t2.get('recall'))}  ({rel_gain(t3.get('recall'), t2.get('recall')):+.2f}%)", "evaluation_metrics.json")
    rk = ev.get("ranking_metrics", {})
    if rk:
        add("Model", "NDCG@10 / MAP (Three-Tower)", f"{f(rk.get('three_tower_ndcg'))} / {f(rk.get('three_tower_map'))}", "evaluation_metrics.json")
    ex = extras()
    if ex:
        ul = ex.get("user_level_paired_tests", {})
        add("Model", "Effect size (Cohen's d, per-customer AUC)", f"{f(ul.get('cohens_d_paired'), 3)} ({ul.get('cohens_d_label', '')})", "evaluation_extras.json")
    return pd.DataFrame(rows, columns=["Section", "Key figure", "Value", "Source file"])
