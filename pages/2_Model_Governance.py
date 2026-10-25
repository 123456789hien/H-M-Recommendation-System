"""
pages/2_Model_Governance.py — Model Governance
================================================================================
The compliance/technical-audit page: architecture, data provenance, and the
5% sampling methodology. Deliberately positioned last in the navigation and
kept plain (no card decoration) — its job is legibility and trust for
whoever needs to check "how was this built", not day-to-day use.
================================================================================
"""

import streamlit as st
import pandas as pd

from utils.data_loader import download_data, load_sampling_report
from utils.kaggle_data import require_kaggle_outputs
from utils import kaggle_data as kd
from utils.theme import inject_global_css, render_sidebar_chrome, thin_rule

st.set_page_config(page_title="Model Governance", page_icon="🗂️", layout="wide")
inject_global_css()
download_data()
require_kaggle_outputs()
render_sidebar_chrome()

st.title("Model Governance")
st.caption("Architecture, data provenance, and sampling methodology — for technical review and compliance.")

st.header("1. Model architecture")
st.markdown(
    """
**Three-Tower Neural Network** — the thesis's central technical contribution:

| Tower | Input | Role |
|---|---|---|
| Tower 1 — Visual | ResNet-50 GAP, 2048-dim | *"What does this product look like?"* |
| Tower 2 — Semantic | BLIP + Sentence-BERT (384) + demographics (3) | *"Does the product's meaning match the customer's profile?"* |
| Tower 3 — Preference Alignment | Hadamard product (item ⊙ user), 10-dim | *"Does this product match what this customer usually prefers?"* |

The **Two-Tower** baseline keeps Tower 1 + Tower 2, **drops Tower 3** — a
controlled experimental design ensuring any measured performance difference
is directly attributable to Tower 3 (thesis Section 3.4.4).

"""
)

_ev, _tm, _wm, _ex = kd.evaluation(), kd.three_metrics(), kd.two_metrics(), kd.extras()
_t3, _t2 = _ev.get("three_tower", {}), _ev.get("two_tower", {})
_tt = _ev.get("statistical_tests", {}).get("paired_ttest", {})
_dpar = (_tm.get("n_parameters") or 0) - (_wm.get("n_parameters") or 0)
_txt = (f"Result: **AUC {kd.f(_t3.get('auc'))} vs {kd.f(_t2.get('auc'))}** ({kd.rel_gain(_t3.get('auc'), _t2.get('auc')):+.2f}%, "
        f"paired t-test p {kd.pval(_tt.get('p_value'))}")
if _ex:
    _ul = _ex.get("user_level_paired_tests", {})
    _txt += f", Cohen's d = {kd.f(_ul.get('cohens_d_paired'), 3)} — {_ul.get('cohens_d_label', '')} effect"
_txt += (f"). The two models differ by **{kd.n(_dpar)} parameters** "
         f"({_dpar / max(_wm.get('n_parameters') or 1, 1) * 100:.2f}% of the Two-Tower): the added third tower and the wider first "
         f"layer of the fusion block that receives it. Total: {kd.n(_tm.get('n_parameters'))} vs {kd.n(_wm.get('n_parameters'))}.")
st.markdown(_txt)
st.caption("Source: evaluation_metrics.json, three_tower_metrics.json, two_tower_metrics.json" + (", evaluation_extras.json" if _ex else ""))

thin_rule()
st.header("2. How the 10 preference topics are found")
st.markdown(
    """
**Ten preference topics** are found automatically by a topic model (LDA) from the
products themselves: **colour, pattern, material and description, product category
and price level**. Every product receives a score on each of the 10 topics.

A customer's **preference profile** is built from what they buy: recent purchases
count more, and each purchase is weighted by its price level, so frequent and
higher-spend purchases shape the profile most.

The **Three-Tower neural network** then combines three signals: how the product
looks (Tower 1), what it means (Tower 2), and how well it matches the customer's
preference profile (Tower 3).

The business reading: for each topic (product line) the console shows **how often
customers buy and how much they spend**, so marketing and range investment go
where customers are most ready to buy.
"""
)

thin_rule()
st.header("3. Data source")
_ov = kd.overview()
st.markdown(
    f"[H&M Personalized Fashion Recommendations Dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) "
    f"(Kaggle): **{kd.n(_ov.get('n_articles'))}** products, **{kd.n(_ov.get('n_customers'))}** customers, "
    f"**{kd.n(_ov.get('n_transactions'))}** transactions ({_ov.get('date_start', 'n/a')} to {_ov.get('date_end', 'n/a')})."
)
st.caption("Source: dataset_overview.json (counted directly from the cleaned files of Sections 1-3)")

thin_rule()
st.header("4. Sampling method for this console's product imagery")
report = load_sampling_report()
st.markdown(
    f"""
Product thumbnails throughout this console use a **{report['sample_rate_target']*100:.0f}%
stratified sample of products per dominant preference** — the same stratified
method and rate as thesis Section 11.2 (which sampled *users*; here it samples
*products*). Each preference's most typical products — the ones that best match the topic's top-20 LDA
features (Section 7-8) — are added so the thumbnails are representative. **Revenue and customer figures are NOT
affected** — those are computed on the full population.
"""
)
col1, col2, col3, col4, col5_ = st.columns(5)
col1.metric("Total products", f"{report['total_population_articles']:,}")
col2.metric("Stratified 5% sample", f"{report['total_sampled_articles_stratified']:,}")
col3.metric("+ topic exemplars", f"+{report.get('extra_articles_exemplars', 0):,}")
col4.metric("+ best sellers", f"+{report.get('extra_articles_bestsellers', 0):,}")
col5_.metric("Total in this prototype", f"{report['total_sampled_articles_final']:,}")
st.caption(
    "The stratified 5% is a random sample, used for the product mix and the slowest sellers. The exemplars show each topic's "
    "typical products; the best sellers are the real top sellers of each topic over the whole catalogue."
)
col5, = st.columns(1)
col5.metric("Mean Absolute Deviation (sample vs. population)", f"{report['mean_absolute_deviation_pp']:.4f}pp")

with st.expander("Full sampling breakdown by segment"):
    per_intent_df = pd.DataFrame(report["per_intention"])
    st.dataframe(
        per_intent_df.rename(columns={
            "intention": "T", "name": "Segment name",
            "population_articles": "Products (population)", "population_share_pct": "Population %",
            "sampled_articles": "Products (sample)", "sample_share_pct": "Sample %",
            "deviation_pp": "Deviation (pp)",
        }),
        use_container_width=True, hide_index=True,
    )

thin_rule()
st.header("5. Known limitations")
st.markdown(
    """
- **Item cold-start** could not be fully evaluated on the current test
  split — see thesis Chapter 6.4.
- Revenue/customer figures on the Command Center are computed on the full
  population; **product imagery** is limited to the 5% demo sample.
- Results reflect **offline evaluation**; no live online A/B testing has
  been conducted.
- **Two non-interchangeable demand metrics are used in this console:**
  the supply-demand gap shown as a 🔴/🟡/🟢 badge (topic_gap_analysis.csv, Section 9) is
  **transaction-count-based** (share of purchase events); the revenue
  trend shown on the Command Center and Segment Detail pages is
  **revenue-weighted** (share of monetary value), computed independently
  from transaction dates. The two differ systematically by segment —
  premium segments (e.g. T8) show a higher revenue share than
  transaction-count share, and low-price segments (e.g. T1) show the
  reverse — because one unit of a premium item contributes more revenue
  than one unit of a basic item. Both are legitimate business lenses; they
  should not be read as the same figure measured at two points in time.
"""
)

with st.expander("Files this app reads (data lineage)"):
    import os
    _rows = []
    for _d, _label in ((os.path.join(kd.DATA_DIR, kd.KO_SUBDIR), "Kaggle outputs"), (kd.population_dir(), "Population analysis")):
        if os.path.isdir(_d):
            for _f in sorted(os.listdir(_d)):
                _rows.append({"Group": _label, "File": _f, "Size (KB)": round(os.path.getsize(os.path.join(_d, _f)) / 1024, 1)})
    if _rows:
        st.dataframe(pd.DataFrame(_rows), hide_index=True, use_container_width=True)
    st.caption("Every number in this console is read from these files; none is typed into the code.")

thin_rule()
st.caption("Source: Master's thesis FQW_DoThiHien_BABDS241 — Deep Learning-Driven "
           "Business Intelligence for Personalized Fashion Retail (HSE, 2026).")
