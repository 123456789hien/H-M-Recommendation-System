"""
pages/1_Data_and_Model.py — everything the Kaggle sections produced, on one page:
key figures, data and customers, the 10 preference topics, how the training data was built, model results.
"""
import streamlit as st

from utils.data_loader import download_data
from utils.kaggle_data import require_kaggle_outputs
from utils.overview import tab_key_figures, tab_data, tab_topics, tab_training, tab_model, render_kpi_strip
from utils.theme import inject_global_css, render_sidebar_chrome

st.set_page_config(page_title="Data and model | H&M", page_icon="📊", layout="wide")
inject_global_css()
download_data()
require_kaggle_outputs()
render_sidebar_chrome()

st.title("Data and model overview")
st.caption("Every figure here is read from a file written by the Kaggle notebooks; the source file is named under each table and chart.")
render_kpi_strip()
tabs = st.tabs(["Key figures", "Data & customers", "Preference topics", "Training data", "Model results"])
with tabs[0]:
    tab_key_figures()
with tabs[1]:
    tab_data()
with tabs[2]:
    tab_topics()
with tabs[3]:
    tab_training()
with tabs[4]:
    tab_model()
