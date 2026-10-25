"""
data_loader.py
================================================================================
Downloads the full demo dataset (5%/preference sample from Kaggle PART 1) from
Google Drive to the Streamlit container's local disk, once per cold
container start, using st.cache_resource.

Why st.cache_resource instead of st.cache_data:
- cache_resource keeps the object (including the fact that files were
  already downloaded to disk) alive for the container's lifetime, without
  re-serializing/re-hashing on every call — a better fit for "download once"
  than for "cache a computed value".
================================================================================
"""

import os
import json
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
from utils.preference_labels import apply_preference_names, preference_name

DATA_DIR = "data"
ZIP_TMP_PATH = "data_export.zip"


def _download_from_drive(file_id: str):
    """Downloads data_export.zip from Google Drive to ZIP_TMP_PATH and returns the path."""
    import gdown
    if not os.path.exists(ZIP_TMP_PATH):
        gdown.download(id=file_id, output=ZIP_TMP_PATH, quiet=False)
    if not os.path.exists(ZIP_TMP_PATH) or os.path.getsize(ZIP_TMP_PATH) < 1_000_000:
        if os.path.exists(ZIP_TMP_PATH):
            os.remove(ZIP_TMP_PATH)                    # a warning page, not the zip: remove it so the next attempt downloads again
        st.error(
            "❌ Zip download failed or the file is too small (Google Drive may "
            "have returned a virus-scan warning page instead of the real file — "
            "this sometimes happens with large public files missing a confirm "
            "token). Double-check GDRIVE_FILE_ID and the sharing permission "
            "(Anyone with the link – Viewer)."
        )
        st.stop()
    return ZIP_TMP_PATH


@st.cache_resource(show_spinner="📥 Downloading demo data from Google Drive (one-time only)...")
def download_data():
    """
    Downloads ONE zip file (data_export.zip, about 60-80 MB: 300 px photos, float16
    features) from Google Drive and extracts it into `data/`. A single zip is far more
    stable than downloading ~5,700 files one by one (gdown.download_folder is prone to
    Google's rate limiting with many files).

    Idempotent: if `data/` already has all the core files within the same
    session/container, skips re-downloading.
    """
    core_files = [
        "articles_sample.csv",
        "article_intention_profiles_sample.csv",
        "article_index_sample.csv",
        "visual_features_sample.npy",
        "semantic_features_sample.npy",
        "demo_personas.csv",
        "intention_labels.json",
        "sampling_report.json",
        "three_tower_best.pt",
        "two_tower_best.pt",
    ]

    already_ok = os.path.isdir(DATA_DIR) and all(
        os.path.exists(os.path.join(DATA_DIR, f)) for f in core_files
    )
    if already_ok:
        return True

    try:
        file_id = st.secrets.get("GDRIVE_FILE_ID", None)
    except Exception:                                   # no secrets configured at all
        file_id = None
    if not file_id or str(file_id).startswith("PASTE_"):
        st.error(
            "⚠️ `GDRIVE_FILE_ID` is not configured in Streamlit Secrets. "
            "See README.md, section 'Deploy', for setup instructions."
        )
        st.stop()

    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = _download_from_drive(str(file_id))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)  # free up disk space right after extraction

    missing = [f for f in core_files if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        st.error(
            f"❌ Extraction incomplete, missing: {missing}. "
            "Check the zip file structure (these files must sit at the zip "
            "root, not nested inside an extra subfolder)."
        )
        st.stop()

    return True


def _plain_persona_label(df: pd.DataFrame) -> pd.DataFrame:
    """Dropdown/table label built from the topic's product-line name, so it never
    depends on the wording of an older data export."""
    if {"dominant_intention", "n_purchases"}.issubset(df.columns) and len(df):
        df["persona_label"] = [
            f"T{int(k)} {preference_name(int(k))} ({int(n)} purchases)"
            for k, n in zip(df["dominant_intention"], df["n_purchases"])
        ]
    return df


@st.cache_data(show_spinner=False)
def load_articles() -> pd.DataFrame:
    download_data()
    df = pd.read_csv(os.path.join(DATA_DIR, "articles_sample.csv"))
    df["article_id"] = df["article_id"].astype(str).str.zfill(10)
    # Defensive: the Kaggle PART 1 union step (5% sample + customer-purchased
    # articles) can duplicate a row if an article was already in the 5%
    # sample AND also purchased by a validation customer. Dedup here fixes
    # any already-uploaded data_export.zip without needing to re-run Kaggle.
    df = df.drop_duplicates(subset="article_id", keep="first").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_article_intention_profiles() -> pd.DataFrame:
    download_data()
    df = pd.read_csv(os.path.join(DATA_DIR, "article_intention_profiles_sample.csv"))
    df["article_id"] = df["article_id"].astype(str).str.zfill(10)
    return df


@st.cache_resource(show_spinner=False)
def load_feature_matrices():
    download_data()
    visual = np.load(os.path.join(DATA_DIR, "visual_features_sample.npy")).astype(np.float32)
    semantic = np.load(os.path.join(DATA_DIR, "semantic_features_sample.npy")).astype(np.float32)
    idx_map = pd.read_csv(os.path.join(DATA_DIR, "article_index_sample.csv"))
    idx_map["article_id"] = idx_map["article_id"].astype(str).str.zfill(10)
    art_feat_idx = dict(zip(idx_map["article_id"], idx_map["feature_index"]))
    return visual, semantic, art_feat_idx


@st.cache_data(show_spinner=False)
def load_demo_personas() -> pd.DataFrame:
    download_data()
    df = pd.read_csv(os.path.join(DATA_DIR, "demo_personas.csv"))
    df["customer_id"] = df["customer_id"].astype(str)
    return _plain_persona_label(df)


@st.cache_data(show_spinner=False)
def load_intention_labels() -> dict:
    """Topic names + colours. Names are the plain product-line names from
    utils/preference_labels.py, whatever text the data export carries."""
    download_data()
    with open(os.path.join(DATA_DIR, "intention_labels.json")) as f:
        return apply_preference_names(json.load(f))


@st.cache_data(show_spinner=False)
def load_sampling_report() -> dict:
    download_data()
    with open(os.path.join(DATA_DIR, "sampling_report.json")) as f:
        report = json.load(f)
    for row in report.get("per_intention", []):
        if "intention" in row:
            row["name"] = preference_name(int(row["intention"]))
    return report


def image_path(article_id: str) -> str:
    """Local image path for a given article_id, once downloaded from Drive."""
    return os.path.join(DATA_DIR, "images", f"{article_id}.jpg")


@st.cache_data(show_spinner=False)
def load_monthly_trends() -> pd.DataFrame:
    """Monthly revenue/units per segment (Kaggle PART 1). Returns an
    empty DataFrame with the right columns if the file isn't present yet —
    so the app degrades gracefully instead of crashing while you re-upload
    an updated data_export.zip to Drive."""
    download_data()
    path = os.path.join(DATA_DIR, "monthly_segment_trends.csv")
    cols = ["year_month", "intention", "revenue", "units", "n_customers", "avg_price", "revenue_share_pct"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    df = df.rename(columns={"dominant_intention": "intention"}) if "dominant_intention" in df.columns else df
    return df


@st.cache_data(show_spinner=False)
def load_customer_validation() -> pd.DataFrame:
    """Real customers (10/preference): their preference profiles, used to build the
    typical customer of each topic in the recommendation studio. From Kaggle PART 1. Empty DataFrame with the right columns if not present
    yet, so the app degrades gracefully."""
    download_data()
    path = os.path.join(DATA_DIR, "customer_validation.csv")
    cols = ["customer_id", "dominant_intention", "confidence", "n_purchases",
            "persona_label", "age", "FN", "Active"] + [f"intention_{k}" for k in range(10)]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    df["customer_id"] = df["customer_id"].astype(str)
    return _plain_persona_label(df)


def model_paths():
    download_data()
    return (
        os.path.join(DATA_DIR, "three_tower_best.pt"),
        os.path.join(DATA_DIR, "two_tower_best.pt"),
    )
