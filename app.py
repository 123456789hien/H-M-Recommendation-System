"""
app.py — H&M Preference Studio
================================================================================
A recommendation app for brands, marketing and merchandising teams, built on the Three-Tower model.

  1. Preference hub        every key figure of the 10 preference topics (price included) and 10 real products per topic
  2. Customers             who prefers what: age, frequency, spend, price level, loyalty by topic
  3. Products and topics   one topic at a time: supply and demand, best and slowest sellers, product mix, what to add or reduce
  4. Recommendation studio pick any product and a customer preference; the Three-Tower model recommends similar products,
                           with match probabilities, the share of each tower, sales, supply and demand, and an add / reduce action

Every number is read from the Kaggle outputs (utils/kaggle_data.py); nothing is typed in.
================================================================================
"""

import streamlit as st

from utils.data_loader import download_data
from utils.kaggle_data import require_kaggle_outputs
from utils.theme import inject_global_css, render_sidebar_chrome

st.set_page_config(page_title="Preference Studio | H&M", page_icon="🧵", layout="wide")
inject_global_css()
download_data()
require_kaggle_outputs()
render_sidebar_chrome()

st.markdown("<div style=\"font-family:Libre Caslon Display, serif; font-size:2.1rem;\">H&amp;M Preference Studio</div>", unsafe_allow_html=True)
st.markdown("**Intention Analytics for fashion brands.** The brand reads what customers prefer across 10 preference topics, then forms its own intention to act: "
            "which product lines to promote, add or reduce, and which products to recommend to which customers. "
            "Built on the Three-Tower model and the full H&M purchase history.")

tabs = st.tabs(["1 · Preference hub", "2 · Customers", "3 · Products and topics", "4 · Recommendation studio"])
with tabs[0]:
    from utils.hub import render_hub
    render_hub()
with tabs[1]:
    from utils.customers import render_customers
    render_customers()
with tabs[2]:
    from utils.products import render_products
    render_products()
with tabs[3]:
    try:
        from utils.studio import render_studio
    except ImportError as e:                      # PyTorch missing on this host: the other tabs keep working
        st.error(f"The recommendation studio needs PyTorch ({e}). Check requirements.txt.")
    else:
        render_studio()
