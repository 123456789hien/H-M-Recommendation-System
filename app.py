# ============================================================================
# H&M FASHION RECOMMENDATION SYSTEM - FINAL INTERACTIVE E-COMMERCE
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
import time
from PIL import Image
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import subprocess

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="H&M Fashion | Smart Recommendations",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'selected_article' not in st.session_state:
    st.session_state.selected_article = None
if 'view_history' not in st.session_state:
    st.session_state.view_history = []

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================
FILE_IDS = {
    'article_metadata.csv': '1RjZmAdpGvQCQHeKpEL30dlTyRenWU1GY',
    'article_intention_profiles.csv': '1aHDWsO8tA2dtKd7bNkk85gk9DP9mNx9M',
    'user_intention_weights.csv': '1C0J3k0FLxCOCxtbLL_dzJ1TDmxWw8rv9',
    'test_interactions.csv': '1AmaZ6DOqTxOOCkpCeRerHz1AyibVoYuG',
    'sampled_user_ids.csv': '1wxbgGcs7K-cUUC8Xm9xEgHyPmXqwE-7w',
    'intention_labels.json': '1Xsw0wM2Wvyo_Mi4PUqfpUYqdDyOEU4bH',
    'user_confidence_scores.csv': '1sa6t6Oun06YpMoJSz7YwN4lufdGYuW6o',
    'customers_cleaned.csv': '1fXH8bSUorehRkbMT2_ROUJUzvBPKzHCO',
    'app_summary.json': '1JJN21tQ4uQ89q-wNvQ1wfnwqV0r7qbHN'
}

IMAGES_FOLDER_ID = "1cj1f09q4OXcBmG5Hpazn_dYrc9kC7qG6"

INTENTION_NAMES = {
    0: "Special Occasion", 1: "Everyday Workwear", 2: "Basics", 3: "Baby Care", 4: "Functional",
    5: "Trendy Casual", 6: "Accessories", 7: "Intimate Care", 8: "Premium Knitwear", 9: "Professional"
}

COLORS = {
    'primary': '#E50010', 'secondary': '#000000', 'bg_light': '#F9F9F9',
    'text_main': '#222222', 'text_muted': '#666666', 'border': '#EEEEEE', 'white': '#FFFFFF'
}

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Montserrat', sans-serif; color: {COLORS['text_main']}; }}
    .main .block-container {{ padding-top: 1rem; max-width: 1200px; }}
    .header-container {{ display: flex; justify-content: center; align-items: center; padding: 1rem 0; margin-bottom: 1rem; border-bottom: 2px solid {COLORS['secondary']}; }}
    .brand-logo {{ font-size: 2.5rem; font-weight: 800; color: {COLORS['primary']}; letter-spacing: -2px; cursor: pointer; }}
    .product-card {{ background: {COLORS['white']}; transition: transform 0.3s ease; position: relative; margin-bottom: 20px; border: 1px solid transparent; }}
    .product-card:hover {{ transform: translateY(-5px); border-color: {COLORS['border']}; }}
    .image-container {{ width: 100%; aspect-ratio: 2/3; overflow: hidden; background: #f0f0f0; position: relative; display: flex; align-items: center; justify-content: center; }}
    .similarity-badge {{ position: absolute; top: 10px; right: 10px; background: rgba(229, 0, 16, 0.9); padding: 4px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; color: white; z-index: 10; }}
    .product-details {{ padding: 10px 0; }}
    .product-title {{ font-size: 0.85rem; font-weight: 600; margin-bottom: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .product-price {{ font-size: 0.95rem; font-weight: 700; color: {COLORS['secondary']}; }}
    .section-title {{ font-size: 1.2rem; font-weight: 700; margin: 1.5rem 0 1rem 0; text-transform: uppercase; letter-spacing: 1px; border-left: 4px solid {COLORS['primary']}; padding-left: 12px; }}
    .detail-container {{ display: flex; gap: 40px; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 40px; }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA ENGINE
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_data_from_drive():
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, 'data')
    images_dir = os.path.join(temp_dir, 'images')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    for filename, file_id in FILE_IDS.items():
        gdown.download(f"https://drive.google.com/uc?id={file_id}", os.path.join(data_dir, filename), quiet=True)
    os.chdir(images_dir)
    subprocess.run(["gdown", f"https://drive.google.com/drive/folders/{IMAGES_FOLDER_ID}", "--folder", "--quiet"], capture_output=True)
    return temp_dir

class RecommendationEngine:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.article_df = pd.read_csv(os.path.join(data_dir, 'data', 'article_metadata.csv'))
        self.article_intentions = pd.read_csv(os.path.join(data_dir, 'data', 'article_intention_profiles.csv'))
        self.user_intentions = pd.read_csv(os.path.join(data_dir, 'data', 'user_intention_weights.csv'))
        self.test_interactions = pd.read_csv(os.path.join(data_dir, 'data', 'test_interactions.csv'))
        self.intention_cols = [f'intention_{i}' for i in range(10)]
        self._build_mappings()
    
    def _build_mappings(self):
        self.user_intent_dict = {str(row['customer_id']): row[self.intention_cols].values.astype(np.float32) for _, row in self.user_intentions.iterrows()}
        self.article_intent_dict = {str(row['article_id']): row[self.intention_cols].values.astype(np.float32) for _, row in self.article_intentions.iterrows()}
        self.user_history = {}
        for _, row in self.test_interactions.iterrows():
            uid, aid = str(row['customer_id']), str(row['article_id'])
            self.user_history.setdefault(uid, []).append(aid)
        self.article_meta_dict = {str(row['article_id']): row.to_dict() for _, row in self.article_df.iterrows()}

    def get_available_users(self):
        # Crucial fix for the AttributeError
        return sorted(list(self.user_intent_dict.keys()))

    def get_user_intention(self, user_id):
        return self.user_intent_dict.get(user_id, np.ones(10) / 10)

    def recommend_for_user(self, user_id, top_n=20):
        user_intent = self.get_user_intention(user_id)
        return self._get_similar_articles(user_intent, top_n)

    def recommend_similar_items(self, article_id, top_n=12):
        article_intent = self.article_intent_dict.get(str(article_id))
        if article_intent is None: return []
        return self._get_similar_articles(article_intent, top_n, exclude_id=str(article_id))

    def _get_similar_articles(self, target_intent, top_n, exclude_id=None):
        article_ids, intentions = [], []
        for aid, intent in self.article_intent_dict.items():
            if aid != exclude_id:
                article_ids.append(aid)
                intentions.append(intent)
        if not intentions: return []
        similarities = cosine_similarity([target_intent], intentions)[0]
        results = sorted(zip(article_ids, similarities), key=lambda x: x[1], reverse=True)[:top_n]
        return results

    def get_article_details(self, article_id):
        return self.article_meta_dict.get(str(article_id))

    def get_article_image_path(self, article_id):
        img_id = str(article_id).zfill(10)
        for root, _, files in os.walk(self.images_dir):
            if f"{img_id}.jpg" in files: return os.path.join(root, f"{img_id}.jpg")
        return None

# ============================================================================
# UI FUNCTIONS
# ============================================================================
def render_product_card(engine, article_id, score=None):
    details = engine.get_article_details(article_id)
    if not details: return
    img_path = engine.get_article_image_path(article_id)
    price = f"{(int(article_id) % 50) + 9.99:.2f}"
    
    with st.container():
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        if score is not None:
            st.markdown(f'<div class="similarity-badge">{score:.0%} MATCH</div>', unsafe_allow_html=True)
        
        if img_path and os.path.exists(img_path):
            st.image(Image.open(img_path), use_container_width=True)
        else:
            st.markdown(f"<div style='color:#ccc;font-size:10px;'>{article_id}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="product-details">
                <div class="product-title">{details.get('prod_name', 'Fashion Item')}</div>
                <div class="product-price">${price}</div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("View Details", key=f"btn_{article_id}", use_container_width=True):
            st.session_state.selected_article = article_id
            if article_id not in st.session_state.view_history:
                st.session_state.view_history.insert(0, article_id)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    try:
        data_path = load_data_from_drive()
        engine = RecommendationEngine(data_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Sidebar
    with st.sidebar:
        st.markdown(f"<h2 style='color:{COLORS['primary']};'>Member Club</h2>", unsafe_allow_html=True)
        users = engine.get_available_users()
        user_id = st.selectbox("Select Customer:", options=users)
        
        if st.session_state.view_history:
            st.markdown("---")
            st.markdown("### Recently Viewed")
            for aid in st.session_state.view_history[:5]:
                d = engine.get_article_details(aid)
                if st.sidebar.button(f"• {d.get('prod_name')[:20]}", key=f"hist_{aid}"):
                    st.session_state.selected_article = aid
                    st.rerun()

    # Header
    st.markdown('<div class="header-container"><div class="brand-logo" onclick="window.location.reload()">H&M</div></div>', unsafe_allow_html=True)

    # Main Content
    if st.session_state.selected_article:
        # Detail View
        article_id = st.session_state.selected_article
        details = engine.get_article_details(article_id)
        if st.button("← Back to Shopping"):
            st.session_state.selected_article = None
            st.rerun()
            
        col1, col2 = st.columns([1, 1.2])
        img_path = engine.get_article_image_path(article_id)
        with col1:
            if img_path: st.image(Image.open(img_path), use_container_width=True)
            else: st.info("No Image")
        with col2:
            st.markdown(f"## {details.get('prod_name')}")
            st.markdown(f"### ${ (int(article_id) % 50) + 9.99:.2f}")
            st.write(f"**Description:** {details.get('detail_desc')}")
            st.button("🛒 ADD TO BAG", use_container_width=True)

        st.markdown("<div class='section-title'>Similar Items You'll Love</div>", unsafe_allow_html=True)
        similars = engine.recommend_similar_items(article_id)
        cols = st.columns(4)
        for idx, (aid, score) in enumerate(similars):
            with cols[idx % 4]: render_product_card(engine, aid, score=score)
    else:
        # Home View
        st.markdown(f"<div class='section-title'>Curated For You</div>", unsafe_allow_html=True)
        recs = engine.recommend_for_user(user_id)
        cols = st.columns(4)
        for idx, (aid, score) in enumerate(recs):
            with cols[idx % 4]: render_product_card(engine, aid, score=score)

if __name__ == "__main__":
    main()
