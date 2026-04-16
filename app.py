# ============================================================================
# H&M FASHION MARKETPLACE - FIXED IMAGE LOADING
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

# Initialize session state
if 'current_intent_vector' not in st.session_state:
    st.session_state.current_intent_vector = np.ones(10) / 10
if 'view_history' not in st.session_state:
    st.session_state.view_history = []
if 'selected_article' not in st.session_state:
    st.session_state.selected_article = None
if 'cart' not in st.session_state:
    st.session_state.cart = []

# ============================================================================
# CONSTANTS
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

COLORS = {
    'primary': '#ee4d2d',
    'secondary': '#fb5533',
    'bg_gray': '#f5f5f5',
    'white': '#ffffff',
    'text': '#222222',
    'text_muted': '#757575',
    'border': '#e8e8e8'
}

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Roboto', sans-serif; background-color: {COLORS['bg_gray']}; }}
    .stApp {{ background-color: {COLORS['bg_gray']}; }}
    
    .product-card {{
        background: white;
        border-radius: 2px;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
        border: 1px solid transparent;
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
    .product-info {{ padding: 8px; }}
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
    .product-price-box {{ display: flex; align-items: baseline; }}
    .currency {{ font-size: 12px; color: {COLORS['primary']}; }}
    .price {{ font-size: 16px; font-weight: 500; color: {COLORS['primary']}; }}
    .sold-count {{ font-size: 10px; color: {COLORS['text_muted']}; margin-top: 4px; }}
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
    .block-container {{ padding-top: 2rem !important; }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_data():
    """Download and extract data from Google Drive"""
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, 'data')
    images_dir = os.path.join(temp_dir, 'images')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # Download data files
    for filename, file_id in FILE_IDS.items():
        gdown.download(f"https://drive.google.com/uc?id={file_id}", os.path.join(data_dir, filename), quiet=True)
    
    # Download images folder - with better error handling
    try:
        os.chdir(images_dir)
        subprocess.run(["gdown", f"https://drive.google.com/drive/folders/{IMAGES_FOLDER_ID}", "--folder", "--quiet"], capture_output=True)
    except Exception as e:
        print(f"Image download warning: {e}")
    
    return temp_dir

# ============================================================================
# MARKETPLACE ENGINE
# ============================================================================
class MarketplaceEngine:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        
        self.article_df = pd.read_csv(os.path.join(data_dir, 'data', 'article_metadata.csv'))
        self.article_intentions = pd.read_csv(os.path.join(data_dir, 'data', 'article_intention_profiles.csv'))
        self.intention_cols = [f'intention_{i}' for i in range(10)]
        self._build_mappings()
    
    def _build_mappings(self):
        self.article_intent_dict = {
            str(row['article_id']): row[self.intention_cols].values.astype(np.float32)
            for _, row in self.article_intentions.iterrows()
        }
        self.article_meta_dict = {
            str(row['article_id']): row.to_dict()
            for _, row in self.article_df.iterrows()
        }

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
        """Find image path - searches all subdirectories"""
        img_id = str(article_id).zfill(10)
        
        # Search recursively
        for root, dirs, files in os.walk(self.images_dir):
            for file in files:
                if file == f"{img_id}.jpg" or file.startswith(img_id):
                    return os.path.join(root, file)
        
        # Try direct path
        direct_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        if os.path.exists(direct_path):
            return direct_path
        
        return None

# ============================================================================
# UI RENDERERS
# ============================================================================
def render_marketplace_header():
    st.markdown(f"""
        <div style="background-color: {COLORS['primary']}; padding: 10px 5%; display: flex; align-items: center; justify-content: space-between; color: white;">
            <div style="font-size: 24px; font-weight: 800;">H&M Marketplace</div>
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
    if not details:
        return
    
    img_path = engine.get_image_path(article_id)
    
    # Placeholder image if not found
    if img_path is None or not os.path.exists(img_path):
        img_url = "https://via.placeholder.com/300x400?text=No+Image"
    else:
        img_url = img_path
    
    price = f"{(int(article_id) % 100) + 19.99:.2f}"
    sold = (int(article_id) % 500) + 10
    
    with st.container():
        st.markdown(f"""
            <div class="product-card">
                <div class="img-box">
        """, unsafe_allow_html=True)
        
        # Display image
        if img_path and os.path.exists(img_path):
            try:
                st.image(Image.open(img_path), use_container_width=True)
            except Exception as e:
                st.image("https://via.placeholder.com/300x400?text=Error", use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x400?text=H&M", use_container_width=True)
            
        st.markdown(f"""
                </div>
                <div class="product-info">
                    <div class="product-name">{details.get('prod_name', 'Unknown')[:50]}</div>
                    <div class="product-price-box">
                        <span class="currency">$</span><span class="price">{price}</span>
                    </div>
                    <div class="sold-count">{sold} sold</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Details", key=f"btn_{article_id}", use_container_width=True):
            clicked_intent = engine.article_intent_dict.get(str(article_id))
            if clicked_intent is not None:
                st.session_state.current_intent_vector = (st.session_state.current_intent_vector * 0.4) + (clicked_intent * 0.6)
            st.session_state.selected_article = article_id
            st.rerun()

def render_detail_view(engine, article_id):
    details = engine.get_article_details(article_id)
    if not details:
        return
    
    if st.button("← Back to Home"):
        st.session_state.selected_article = None
        st.rerun()
        
    col1, col2 = st.columns([1, 1.2])
    img_path = engine.get_image_path(article_id)
    price = f"{(int(article_id) % 100) + 19.99:.2f}"
    
    with col1:
        if img_path and os.path.exists(img_path):
            try:
                st.image(Image.open(img_path), use_container_width=True)
            except:
                st.image("https://via.placeholder.com/300x400?text=H&M", use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x400?text=H&M", use_container_width=True)
        
    with col2:
        st.markdown(f"# {details.get('prod_name', 'Unknown')}")
        st.markdown(f"<h2 style='color:{COLORS['primary']}'>${price}</h2>", unsafe_allow_html=True)
        st.markdown("---")
        st.write(f"**Category:** {details.get('product_type_name', 'N/A')}")
        st.write(f"**Group:** {details.get('product_group_name', 'N/A')}")
        st.write(f"**Description:** {details.get('detail_desc', 'No description')}")
        
        st.markdown("### 🚚 Shipping")
        st.caption("Free shipping for orders over $50")
        
        st.button("🛒 ADD TO CART", use_container_width=True)
        st.button("BUY NOW", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### YOU MAY ALSO LIKE")
    similars = engine.get_recommendations(engine.article_intent_dict[str(article_id)], top_n=12)
    cols = st.columns(6)
    for idx, (aid, _) in enumerate(similars[1:]):
        with cols[idx % 6]:
            render_product_card(engine, aid)

# ============================================================================
# MAIN
# ============================================================================
def main():
    try:
        data_path = load_data()
        engine = MarketplaceEngine(data_path)
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return

    render_marketplace_header()

    if st.session_state.selected_article:
        render_detail_view(engine, st.session_state.selected_article)
    else:
        # Banner
        st.markdown('<div class="banner">H&M NEW SEASON: UP TO 50% OFF</div>', unsafe_allow_html=True)
        
        # Categories
        st.markdown("### CATEGORIES")
        cat_cols = st.columns(10)
        for i in range(10):
            with cat_cols[i]:
                if st.button(f"{INTENTION_NAMES[i]}", key=f"cat_{i}", use_container_width=True):
                    new_intent = np.zeros(10)
                    new_intent[i] = 1.0
                    st.session_state.current_intent_vector = new_intent
                    st.rerun()

        # Personalized Feed
        st.markdown("### DAILY DISCOVER")
        recs = engine.get_recommendations(st.session_state.current_intent_vector, top_n=30)
        
        cols = st.columns(5)
        for idx, (aid, _) in enumerate(recs):
            with cols[idx % 5]:
                render_product_card(engine, aid)

if __name__ == "__main__":
    main()
