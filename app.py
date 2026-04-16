# ============================================================================
# H&M FASHION RECOMMENDATION SYSTEM - REFINED E-COMMERCE UI
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
    page_title="H&M Fashion | Personal Shopping",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    0: "Special Occasion Dressing",
    1: "Everyday Workwear Comfort",
    2: "Utilitarian Basics",
    3: "Infant & Nurturing Care",
    4: "Functional Versatility",
    5: "Trendy & Casual Provisioning",
    6: "Hedonic Accessories",
    7: "Intimate Self-Care",
    8: "Premium Quality Investment",
    9: "Professional Identity Expression"
}

INTENTION_ICONS = {
    0: "👗", 1: "👕", 2: "🧦", 3: "👶", 4: "👖",
    5: "🧥", 6: "👜", 7: "💕", 8: "🧶", 9: "👔"
}

INTENTION_DESCRIPTIONS = {
    0: "Dresses and jumpsuits for parties, events, and celebrations. Trend-led, fashion-forward pieces.",
    1: "Shirts and blouses for daily professional wear. Comfortable, practical, and style-conscious.",
    2: "Socks, tights, and basic items. Functional replenishment with minimal deliberation.",
    3: "Baby jumpsuits and infant wear. Caregiving purchases driven by nurturing affect.",
    4: "Trousers and jeans selected for practical attributes like pockets and durability.",
    5: "Hoodies and sweaters for children. Balancing child preferences with cost.",
    6: "Scarves, bags, shoes, and accessories. Novelty-seeking and hedonic stimulation.",
    7: "Bras and underwear bottoms. Personal well-being and bodily comfort.",
    8: "Sweaters and knitwear. Investment in premium quality and aesthetic experience.",
    9: "Shirts for men's professional wear. Workplace role-normative purchases."
}

COLORS = {
    'primary': '#E50010',
    'secondary': '#000000',
    'bg_light': '#F9F9F9',
    'text_main': '#222222',
    'text_muted': '#666666',
    'border': '#EEEEEE',
    'white': '#FFFFFF',
    'success': '#198754'
}

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Montserrat', sans-serif;
        color: {COLORS['text_main']};
    }}

    .main .block-container {{
        padding-top: 1rem;
        max-width: 1200px;
    }}

    /* Header Styling */
    .header-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 2px solid {COLORS['secondary']};
    }}
    .brand-logo {{
        font-size: 3rem;
        font-weight: 800;
        color: {COLORS['primary']};
        letter-spacing: -2px;
    }}

    /* Product Grid */
    .product-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 30px;
    }}

    /* Product Card */
    .product-card {{
        background: {COLORS['white']};
        transition: transform 0.3s ease;
        position: relative;
        margin-bottom: 20px;
    }}
    .product-card:hover {{
        transform: translateY(-5px);
    }}
    .image-container {{
        width: 100%;
        aspect-ratio: 2/3;
        overflow: hidden;
        background: #f0f0f0;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .similarity-badge {{
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(229, 0, 16, 0.9);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        color: white;
        z-index: 10;
    }}
    .product-details {{
        padding: 12px 0;
    }}
    .product-title {{
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .product-meta {{
        font-size: 0.75rem;
        color: {COLORS['text_muted']};
        margin-bottom: 8px;
        text-transform: capitalize;
    }}
    .product-price {{
        font-size: 1rem;
        font-weight: 700;
        color: {COLORS['secondary']};
    }}
    .product-tag {{
        display: inline-block;
        font-size: 10px;
        padding: 2px 6px;
        background: {COLORS['bg_light']};
        border: 1px solid {COLORS['border']};
        margin-right: 4px;
        margin-top: 4px;
        border-radius: 2px;
    }}

    /* Section Headers */
    .section-title {{
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-left: 5px solid {COLORS['primary']};
        padding-left: 15px;
    }}

    /* Sidebar Customization */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['white']};
        border-right: 1px solid {COLORS['border']};
    }}
    
    /* Stats Card */
    .stat-box {{
        padding: 1.5rem;
        background: {COLORS['bg_light']};
        border: 1px solid {COLORS['border']};
        text-align: center;
    }}
    .stat-val {{
        font-size: 1.8rem;
        font-weight: 800;
        color: {COLORS['primary']};
    }}
    .stat-label {{
        font-size: 0.7rem;
        text-transform: uppercase;
        color: {COLORS['text_muted']};
        margin-top: 5px;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_data_from_drive():
    progress_container = st.empty()
    with progress_container.container():
        st.markdown(f"""
            <div style="text-align: center; padding: 3rem;">
                <h2 style="color: {COLORS['primary']};">H&M</h2>
                <p>Initializing Boutique Experience...</p>
            </div>
        """, unsafe_allow_html=True)
        progress_bar = st.progress(0)
    
    try:
        temp_dir = tempfile.mkdtemp()
        data_dir = os.path.join(temp_dir, 'data')
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        for i, (filename, file_id) in enumerate(FILE_IDS.items()):
            dest_path = os.path.join(data_dir, filename)
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, dest_path, quiet=True)
            progress_bar.progress(10 + int(i / len(FILE_IDS) * 40))
        
        # Extract images from folder
        os.chdir(images_dir)
        folder_url = f"https://drive.google.com/drive/folders/{IMAGES_FOLDER_ID}"
        subprocess.run(["gdown", folder_url, "--folder", "--quiet"], capture_output=True)
        
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_container.empty()
        return temp_dir
    except Exception as e:
        progress_container.empty()
        st.error(f"Initialization failed: {str(e)}")
        raise e

# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================
class RecommendationEngine:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        
        self.article_df = pd.read_csv(os.path.join(data_dir, 'data', 'article_metadata.csv'))
        self.article_intentions = pd.read_csv(os.path.join(data_dir, 'data', 'article_intention_profiles.csv'))
        self.user_intentions = pd.read_csv(os.path.join(data_dir, 'data', 'user_intention_weights.csv'))
        self.test_interactions = pd.read_csv(os.path.join(data_dir, 'data', 'test_interactions.csv'))
        
        try:
            with open(os.path.join(data_dir, 'data', 'app_summary.json'), 'r') as f:
                self.app_summary = json.load(f)
        except: self.app_summary = {}
            
        self.intention_cols = [f'intention_{i}' for i in range(10)]
        self._build_mappings()
    
    def _build_mappings(self):
        self.user_intent_dict = {
            str(row['customer_id']): row[self.intention_cols].values.astype(np.float32)
            for _, row in self.user_intentions.iterrows()
        }
        self.article_intent_dict = {
            str(row['article_id']): row[self.intention_cols].values.astype(np.float32)
            for _, row in self.article_intentions.iterrows()
        }
        self.user_history = {}
        for _, row in self.test_interactions.iterrows():
            uid, aid = str(row['customer_id']), str(row['article_id'])
            self.user_history.setdefault(uid, []).append(aid)
        
        self.article_meta_dict = {
            str(row['article_id']): row.to_dict()
            for _, row in self.article_df.iterrows()
        }

    def get_user_intention(self, user_id):
        return self.user_intent_dict.get(user_id, np.ones(10) / 10)

    def get_user_purchased_articles(self, user_id):
        return self.user_history.get(user_id, [])

    def recommend_by_intention(self, user_id, top_n=24):
        user_intent = self.get_user_intention(user_id)
        purchased = set(self.get_user_purchased_articles(user_id))
        
        article_ids, intentions = [], []
        for aid, intent in self.article_intent_dict.items():
            if aid not in purchased:
                article_ids.append(aid)
                intentions.append(intent)
        
        if not article_ids: return []
        
        similarities = cosine_similarity([user_intent], intentions)[0]
        results = sorted(zip(article_ids, similarities), key=lambda x: x[1], reverse=True)[:top_n]
        return results

    def get_article_details(self, article_id):
        return self.article_meta_dict.get(str(article_id))

    def get_article_image_path(self, article_id):
        # H&M article IDs are 10 digits, often starting with 0
        img_id = str(article_id).zfill(10)
        # Search recursively in images_dir
        for root, dirs, files in os.walk(self.images_dir):
            # Check for direct match or filename starting with ID
            if f"{img_id}.jpg" in files:
                return os.path.join(root, f"{img_id}.jpg")
            # Some datasets might have subfolders like '010', '011' based on first 3 digits
            prefix = img_id[:3]
            sub_path = os.path.join(root, prefix, f"{img_id}.jpg")
            if os.path.exists(sub_path):
                return sub_path
        return None

    def get_available_users(self):
        return sorted(list(self.user_intent_dict.keys()))

# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_header():
    st.markdown(f"""
        <div class="header-container">
            <div class="brand-logo">H&M</div>
        </div>
    """, unsafe_allow_html=True)

def render_product_card(engine, article_id, score=None, is_history=False):
    details = engine.get_article_details(article_id)
    if not details: return
    
    img_path = engine.get_article_image_path(article_id)
    price = f"{(int(article_id) % 50) + 9.99:.2f}"
    
    st.markdown('<div class="product-card">', unsafe_allow_html=True)
    
    # Image Section
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    if score is not None:
        st.markdown(f'<div class="similarity-badge">{score:.0%} MATCH</div>', unsafe_allow_html=True)
    elif is_history:
        st.markdown(f'<div class="similarity-badge" style="background:#000;">PURCHASED</div>', unsafe_allow_html=True)
    
    if img_path and os.path.exists(img_path):
        st.image(Image.open(img_path), use_container_width=True)
    else:
        # Fallback if image not found
        st.markdown(f"<div style='text-align:center;color:#ccc;'>No Image<br><small>{article_id}</small></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Details Section
    name = details.get('prod_name', 'Fashion Item')
    cat = details.get('product_type_name', 'Apparel')
    color = details.get('colour_group_name', 'Mixed')
    
    st.markdown(f"""
        <div class="product-details">
            <div class="product-title">{name}</div>
            <div class="product-meta">{color} | {cat}</div>
            <div class="product-price">${price}</div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("View Details"):
        st.write(f"**Description:** {details.get('detail_desc', 'Premium H&M quality.')}")
        st.write(f"**Group:** {details.get('product_group_name', 'N/A')}")
        st.write(f"**Article ID:** `{article_id}`")
    st.markdown('</div>', unsafe_allow_html=True)

def render_for_you(engine, user_id):
    st.markdown(f"<div class='section-title'>Recommended For You</div>", unsafe_allow_html=True)
    
    with st.spinner("Finding matches..."):
        recommendations = engine.recommend_by_intention(user_id, top_n=20)
    
    if recommendations:
        cols = st.columns(4)
        for idx, (aid, score) in enumerate(recommendations):
            with cols[idx % 4]:
                render_product_card(engine, aid, score=score)
    else:
        st.info("No recommendations found for this user.")

def render_history(engine, user_id):
    st.markdown(f"<div class='section-title'>Your Purchase History</div>", unsafe_allow_html=True)
    purchases = engine.get_user_purchased_articles(user_id)
    
    if purchases:
        st.write(f"Showing {len(purchases)} past purchases")
        cols = st.columns(4)
        for idx, aid in enumerate(reversed(purchases)): # Show newest first
            with cols[idx % 4]:
                render_product_card(engine, aid, is_history=True)
    else:
        st.info("No purchase history found.")

def render_dna(engine, user_id):
    st.markdown("<div class='section-title'>Your Style DNA</div>", unsafe_allow_html=True)
    user_intent = engine.get_user_intention(user_id)
    categories = [INTENTION_NAMES[i] for i in range(10)]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=user_intent.tolist(), theta=categories, fill='toself',
        fillcolor='rgba(229, 0, 16, 0.2)', line=dict(color='#E50010', width=2)
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MAIN
# ============================================================================
def main():
    render_header()
    
    try:
        data_path = load_data_from_drive()
        engine = RecommendationEngine(data_path)
        
        with st.sidebar:
            st.markdown(f"<h2 style='color:{COLORS['primary']};'>Member Login</h2>", unsafe_allow_html=True)
            
            # Login Options
            login_mode = st.radio("Login Method", ["Select from List", "Enter Customer ID"])
            
            if login_mode == "Select from List":
                users = engine.get_available_users()
                selected_user = st.selectbox("Choose a Customer:", options=users)
            else:
                selected_user = st.text_input("Enter Customer ID:", value=engine.get_available_users()[0])
            
            st.markdown("---")
            st.markdown("### 📊 Account Stats")
            history_count = len(engine.get_user_purchased_articles(selected_user))
            st.metric("Total Purchases", history_count)
            
            st.markdown("---")
            st.caption(f"Model: Three-Tower NN")
            st.caption(f"AUC: {engine.app_summary.get('model_performance', {}).get('three_tower_auc', 0.8201):.4f}")
        
        # Tabs for Content
        tab1, tab2, tab3 = st.tabs(["✨ FOR YOU", "📜 HISTORY", "🧬 STYLE DNA"])
        
        with tab1: render_for_you(engine, selected_user)
        with tab2: render_history(engine, selected_user)
        with tab3: render_dna(engine, selected_user)
        
        st.markdown(f"<div style='text-align:center;margin-top:50px;color:#888;font-size:0.8rem;'>© 2026 H&M Fashion AI System</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error("Application Error. Please refresh.")
        st.exception(e)

if __name__ == "__main__":
    main()
