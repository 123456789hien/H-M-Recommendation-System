# ============================================================================
# H&M FASHION
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
    page_title="H&M Fashion | Shop the Latest Trends",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for the Guest experience
if 'current_intent_vector' not in st.session_state:
    # Default intent vector (balanced/neutral)
    st.session_state.current_intent_vector = np.ones(10) / 10
if 'view_history' not in st.session_state:
    st.session_state.view_history = []
if 'selected_article' not in st.session_state:
    st.session_state.selected_article = None
if 'cart' not in st.session_state:
    st.session_state.cart = []

# ============================================================================
# CONSTANTS & COLORS (Shopee/Lazada Inspired)
# ============================================================================
FILE_IDS = {
    'article_metadata.csv': '1RjZmAdpGvQCQHeKpEL30dlTyRenWU1GY',
    'article_intention_profiles.csv': '1aHDWsO8tA2dtKd7bNkk85gk9DP9mNx9M',
    'user_intention_weights.csv': '1C0J3k0FLxCOCxtbLL_dzJ1TDmxWw8rv9',
    'test_interactions.csv': '1AmaZ6DOqTxOOCkpCeRerHz1AyibVoYuG',
    'intention_labels.json': '1Xsw0wM2Wvyo_Mi4PUqfpUYqdDyOEU4bH',
    'app_summary.json': '1JJN21tQ4uQ89q-wNvQ1wfnwqV0r7qbHN'
}

IMAGES_FOLDER_ID = "1cj1f09q4OXcBmG5Hpazn_dYrc9kC7qG6"

INTENTION_NAMES = {
    0: "Party", 1: "Workwear", 2: "Basics", 3: "Baby", 4: "Denim",
    5: "Kids", 6: "Accessories", 7: "Lingerie", 8: "Knitwear", 9: "Men"
}

# Shopee-like Orange/Red Palette
COLORS = {
    'primary': '#ee4d2d', # Shopee Orange
    'secondary': '#fb5533',
    'bg_gray': '#f5f5f5',
    'white': '#ffffff',
    'text': '#222222',
    'text_muted': '#757575',
    'border': '#e8e8e8'
}

# ============================================================================
# CUSTOM CSS (Marketplace UI)
# ============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Roboto', sans-serif;
        background-color: {COLORS['bg_gray']};
    }}

    /* Shopee Header */
    .stApp {{ background-color: {COLORS['bg_gray']}; }}
    
    .header-bar {{
        background: linear-gradient(-180deg,#f53d2d,#f63);
        padding: 15px 10%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        height: 80px;
    }}
    
    .search-container {{
        flex-grow: 1;
        margin: 0 40px;
        position: relative;
    }}
    
    .search-input {{
        width: 100%;
        padding: 10px 15px;
        border-radius: 2px;
        border: none;
        font-size: 14px;
    }}

    /* Product Grid Lazada Style */
    .product-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
        gap: 12px;
        padding: 20px 0;
    }}

    .product-card {{
        background: white;
        border-radius: 2px;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
        border: 1px solid transparent;
        position: relative;
    }}
    
    .product-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 2px 12px rgba(0,0,0,0.12);
        border-color: {COLORS['primary']};
    }}

    .img-box {{
        width: 100%;
        aspect-ratio: 1/1;
        background: #f8f8f8;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }}

    .product-info {{
        padding: 8px;
    }}

    .product-name {{
        font-size: 12px;
        line-height: 14px;
        height: 28px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        margin-bottom: 8px;
        color: {COLORS['text']};
    }}

    .product-price-box {{
        display: flex;
        align-items: baseline;
    }}

    .currency {{ font-size: 12px; color: {COLORS['primary']}; }}
    .price {{ font-size: 16px; font-weight: 500; color: {COLORS['primary']}; }}
    
    .sold-count {{
        font-size: 10px;
        color: {COLORS['text_muted']};
        margin-top: 4px;
    }}

    /* Banner */
    .banner {{
        width: 100%;
        height: 300px;
        background: linear-gradient(90deg, #ff6a00, #ee0979);
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 20px;
    }}

    /* Category Icon */
    .cat-item {{
        text-align: center;
        padding: 10px;
        background: white;
        border: 1px solid {COLORS['border']};
        cursor: pointer;
    }}
    .cat-item:hover {{ color: {COLORS['primary']}; }}

    /* Fix Streamlit spacing */
    .block-container {{ padding-top: 2rem !important; }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA ENGINE (Cold Start Optimized)
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_data():
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

class MarketplaceEngine:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.article_df = pd.read_csv(os.path.join(data_dir, 'data', 'article_metadata.csv'))
        self.article_intentions = pd.read_csv(os.path.join(data_dir, 'data', 'article_intention_profiles.csv'))
        self.intention_cols = [f'intention_{i}' for i in range(10)]
        self._build_mappings()
    
    def _build_mappings(self):
        self.article_intent_dict = {str(row['article_id']): row[self.intention_cols].values.astype(np.float32) for _, row in self.article_intentions.iterrows()}
        self.article_meta_dict = {str(row['article_id']): row.to_dict() for _, row in self.article_df.iterrows()}

    def get_recommendations(self, intent_vector, top_n=30):
        article_ids, intentions = [], []
        for aid, intent in self.article_intent_dict.items():
            article_ids.append(aid)
            intentions.append(intent)
        
        similarities = cosine_similarity([intent_vector], intentions)[0]
        results = sorted(zip(article_ids, similarities), key=lambda x: x[1], reverse=True)[:top_n]
        return results

    def get_article_details(self, article_id):
        return self.article_meta_dict.get(str(article_id))

    def get_image_path(self, article_id):
        img_id = str(article_id).zfill(10)
        for root, _, files in os.walk(self.images_dir):
            if f"{img_id}.jpg" in files: return os.path.join(root, f"{img_id}.jpg")
        return None

# ============================================================================
# UI RENDERERS
# ============================================================================
def render_marketplace_header():
    # Simulated Shopee Header
    st.markdown(f"""
        <div style="background-color: {COLORS['primary']}; padding: 10px 5%; display: flex; align-items: center; justify-content: space-between; color: white;">
            <div style="font-size: 24px; font-weight: 800; cursor: pointer;" onclick="window.location.reload()">H&M Marketplace</div>
            <div style="flex-grow: 1; margin: 0 30px;">
                <input type="text" placeholder="Search for fashion..." style="width: 100%; padding: 8px 15px; border: none; border-radius: 2px;">
            </div>
            <div style="display: flex; gap: 20px; align-items: center;">
                <span>Login</span>
                <span>Signup</span>
                <span style="font-size: 20px;">🛒</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_product_card(engine, article_id, score=None):
    details = engine.get_article_details(article_id)
    if not details: return
    
    img_path = engine.get_image_path(article_id)
    price = f"{(int(article_id) % 100) + 19.99:.2f}"
    sold = (int(article_id) % 500) + 10
    
    with st.container():
        st.markdown(f"""
            <div class="product-card">
                <div class="img-box">
        """, unsafe_allow_html=True)
        
        if img_path and os.path.exists(img_path):
            st.image(Image.open(img_path), use_container_width=True)
        else:
            st.markdown("<div style='color:#ccc; font-size:10px;'>No Image</div>", unsafe_allow_html=True)
            
        st.markdown(f"""
                </div>
                <div class="product-info">
                    <div class="product-name">{details.get('prod_name')}</div>
                    <div class="product-price-box">
                        <span class="currency">$</span><span class="price">{price}</span>
                    </div>
                    <div class="sold-count">{sold} sold</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Details", key=f"btn_{article_id}", use_container_width=True):
            # Update user intent based on clicked item (Bayesian Cold Start Logic)
            clicked_intent = engine.article_intent_dict.get(str(article_id))
            if clicked_intent is not None:
                # Bayesian Update: Mix current intent with clicked intent
                st.session_state.current_intent_vector = (st.session_state.current_intent_vector * 0.4) + (clicked_intent * 0.6)
            
            st.session_state.selected_article = article_id
            st.rerun()

def render_detail_view(engine, article_id):
    details = engine.get_article_details(article_id)
    if not details: return
    
    if st.button("← Back to Home"):
        st.session_state.selected_article = None
        st.rerun()
        
    col1, col2 = st.columns([1, 1.2])
    img_path = engine.get_image_path(article_id)
    price = f"{(int(article_id) % 100) + 19.99:.2f}"
    
    with col1:
        if img_path: st.image(Image.open(img_path), use_container_width=True)
        else: st.info("Image not available")
        
    with col2:
        st.markdown(f"# {details.get('prod_name')}")
        st.markdown(f"<h2 style='color:{COLORS['primary']}'>${price}</h2>", unsafe_allow_html=True)
        st.markdown("---")
        st.write(f"**Category:** {details.get('product_type_name')}")
        st.write(f"**Group:** {details.get('product_group_name')}")
        st.write(f"**Description:** {details.get('detail_desc')}")
        
        st.markdown("### 🚚 Shipping")
        st.caption("Free shipping for orders over $50")
        
        st.button("🛒 ADD TO CART", use_container_width=True)
        st.button("BUY NOW", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### YOU MAY ALSO LIKE")
    # Item-to-Item recommendations
    similars = engine.get_recommendations(engine.article_intent_dict[str(article_id)], top_n=12)
    cols = st.columns(6)
    for idx, (aid, _) in enumerate(similars[1:]): # skip self
        with cols[idx % 6]:
            render_product_card(engine, aid)

def main():
    try:
        data_path = load_data()
        engine = MarketplaceEngine(data_path)
    except:
        st.error("Data loading failed.")
        return

    render_marketplace_header()

    if st.session_state.selected_article:
        render_detail_view(engine, st.session_state.selected_article)
    else:
        # Home Page Components
        # 1. Banner
        st.markdown('<div class="banner">H&M NEW SEASON: UP TO 50% OFF</div>', unsafe_allow_html=True)
        
        # 2. Categories
        st.markdown("### CATEGORIES")
        cat_cols = st.columns(10)
        for i in range(10):
            with cat_cols[i]:
                if st.button(f"{INTENTION_NAMES[i]}", key=f"cat_{i}", use_container_width=True):
                    # Filter by intent
                    new_intent = np.zeros(10)
                    new_intent[i] = 1.0
                    st.session_state.current_intent_vector = new_intent
                    st.rerun()

        # 3. Personalized Feed (Cold Start)
        st.markdown("### DAILY DISCOVER")
        recs = engine.get_recommendations(st.session_state.current_intent_vector, top_n=30)
        
        cols = st.columns(5)
        for idx, (aid, _) in enumerate(recs):
            with cols[idx % 5]:
                render_product_card(engine, aid)

if __name__ == "__main__":
    main()
