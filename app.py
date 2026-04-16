# ============================================================================
# H&M FASHION RECOMMENDATION SYSTEM - PROFESSIONAL E-COMMERCE UI
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
# CUSTOM CSS FOR E-COMMERCE UI
# ============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Montserrat', sans-serif;
        color: {COLORS['text_main']};
    }}

    .main .block-container {{
        padding-top: 2rem;
        max-width: 1200px;
    }}

    /* Header Styling */
    .header-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 2px solid {COLORS['secondary']};
    }}
    .brand-logo {{
        font-size: 2.5rem;
        font-weight: 800;
        color: {COLORS['primary']};
        letter-spacing: -2px;
    }}
    .header-nav {{
        display: flex;
        gap: 20px;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.9rem;
    }}

    /* Product Grid */
    .product-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 30px;
        margin-top: 20px;
    }}

    /* Product Card */
    .product-card {{
        background: {COLORS['white']};
        transition: transform 0.3s ease;
        position: relative;
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
    }}
    .similarity-badge {{
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(255, 255, 255, 0.9);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        color: {COLORS['primary']};
        border: 1px solid {COLORS['primary']};
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
    }}

    /* Sidebar Customization */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['white']};
        border-right: 1px solid {COLORS['border']};
    }}
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
        font-size: 14px;
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
    """Download all data files and images from Google Drive"""
    progress_container = st.empty()
    with progress_container.container():
        st.markdown(f"""
            <div style="text-align: center; padding: 3rem;">
                <h2 style="color: {COLORS['primary']};">H&M</h2>
                <p>Initializing Boutique Experience...</p>
                <div style="font-size: 0.8rem; color: #888;">Fetching latest collections from secure servers</div>
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
            with open(os.path.join(data_dir, 'data', 'intention_labels.json'), 'r') as f:
                self.intention_labels = json.load(f)
        except:
            self.intention_labels = {}
            
        try:
            self.user_confidence_df = pd.read_csv(os.path.join(data_dir, 'data', 'user_confidence_scores.csv'))
        except:
            self.user_confidence_df = pd.DataFrame()
            
        try:
            with open(os.path.join(data_dir, 'data', 'app_summary.json'), 'r') as f:
                self.app_summary = json.load(f)
        except:
            self.app_summary = {}
            
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
        
        self.user_confidence = {
            str(row['customer_id']): row.get('confidence', 0.5)
            for _, row in self.user_confidence_df.iterrows()
        } if not self.user_confidence_df.empty else {}
        
        self.article_meta_dict = {
            str(row['article_id']): row.to_dict()
            for _, row in self.article_df.iterrows()
        }

    def get_user_intention(self, user_id):
        return self.user_intent_dict.get(user_id, np.ones(10) / 10)

    def get_user_confidence(self, user_id):
        return self.user_confidence.get(user_id, 0.0)

    def get_user_purchased_articles(self, user_id):
        return self.user_history.get(user_id, [])

    def recommend_by_intention(self, user_id, top_n=24, exclude_purchased=True):
        user_intent = self.get_user_intention(user_id)
        purchased = set(self.get_user_purchased_articles(user_id)) if exclude_purchased else set()
        
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

    def get_intention_description(self, intention_id):
        return {
            "name": INTENTION_NAMES.get(intention_id, f'Intention {intention_id}'),
            "description": INTENTION_DESCRIPTIONS.get(intention_id, 'Shopping preference category'),
            "icon": INTENTION_ICONS.get(intention_id, "🎯")
        }

    def get_article_image_path(self, article_id):
        img_id = str(article_id).zfill(10)
        for root, dirs, files in os.walk(self.images_dir):
            for file in files:
                if file.startswith(img_id) or file == f"{img_id}.jpg":
                    return os.path.join(root, file)
        return None

    def get_available_users(self):
        return sorted(list(self.user_history.keys()))

    def get_articles_by_intention(self, intention_id, top_n=24):
        articles = []
        for aid, intent_vec in self.article_intent_dict.items():
            if np.argmax(intent_vec) == intention_id:
                articles.append((aid, intent_vec[intention_id]))
        articles.sort(key=lambda x: x[1], reverse=True)
        return articles[:top_n]

# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_header():
    st.markdown(f"""
        <div class="header-container">
            <div class="brand-logo">H&M</div>
            <div class="header-nav">
                <span>Ladies</span>
                <span>Men</span>
                <span>Divided</span>
                <span>Baby</span>
                <span>Kids</span>
                <span>H&M HOME</span>
                <span style="color: {COLORS['primary']};">Sale</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_product_card(engine, article_id, score=None, show_tag=True):
    details = engine.get_article_details(article_id)
    if not details: return
    
    img_path = engine.get_article_image_path(article_id)
    
    # Generate a fake price based on article_id for realistic look
    price = f"{(int(article_id) % 50) + 9.99:.2f}"
    
    st.markdown('<div class="product-card">', unsafe_allow_html=True)
    
    # Image Section
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    if score is not None:
        st.markdown(f'<div class="similarity-badge">{score:.0%} MATCH</div>', unsafe_allow_html=True)
    
    if img_path:
        try:
            img = Image.open(img_path)
            st.image(img, use_container_width=True)
        except:
            st.image("https://via.placeholder.com/300x450?text=H&M+Fashion", use_container_width=True)
    else:
        st.image("https://via.placeholder.com/300x450?text=H&M+Fashion", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Details Section
    name = details.get('prod_name', 'Unknown Product')
    cat = details.get('product_type_name', 'Fashion')
    group = details.get('product_group_name', 'Apparel')
    color = details.get('colour_group_name', 'Mixed')
    desc = details.get('detail_desc', 'Premium quality H&M fashion piece.')
    
    st.markdown(f"""
        <div class="product-details">
            <div class="product-title">{name}</div>
            <div class="product-meta">{color} | {cat}</div>
            <div class="product-price">${price}</div>
            <div style="margin-top: 8px;">
    """, unsafe_allow_html=True)
    
    if show_tag:
        article_intent = engine.article_intent_dict.get(str(article_id), np.zeros(10))
        top_intent = np.argmax(article_intent)
        intent_info = engine.get_intention_description(top_intent)
        st.markdown(f"<span class='product-tag'>{intent_info['icon']} {intent_info['name']}</span>", unsafe_allow_html=True)
    
    # Tooltip-like description on hover (simplified for Streamlit)
    st.markdown('</div></div></div>', unsafe_allow_html=True)
    with st.expander("Product Details"):
        st.caption(f"**Group:** {group}")
        st.caption(f"**Description:** {desc}")

def render_for_you(engine, user_id):
    if not user_id:
        st.warning("Please sign in from the sidebar to see your personalized shop.")
        return
    
    user_intent = engine.get_user_intention(user_id)
    dom_idx = np.argmax(user_intent)
    intent_info = engine.get_intention_description(dom_idx)
    confidence = engine.get_user_confidence(user_id)
    
    st.markdown(f"<div class='section-title'>Curated for your style: {intent_info['name']}</div>", unsafe_allow_html=True)
    
    # Stats row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{intent_info["icon"]}</div><div class="stat-label">Style Profile</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{confidence:.0%}</div><div class="stat-label">Profile Accuracy</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{len(engine.get_user_purchased_articles(user_id))}</div><div class="stat-label">Items Purchased</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.spinner("Analyzing your fashion intent..."):
        recommendations = engine.recommend_by_intention(user_id, top_n=24)
    
    if recommendations:
        cols = st.columns(4)
        for idx, (aid, score) in enumerate(recommendations):
            with cols[idx % 4]:
                render_product_card(engine, aid, score=score)
    else:
        st.info("No items match your profile currently. Try exploring our new arrivals!")

def render_explore(engine):
    st.markdown("<div class='section-title'>Shop by Category</div>", unsafe_allow_html=True)
    
    # Intention Selector
    selected_int = st.selectbox(
        "What are you looking for today?",
        options=range(10),
        format_func=lambda x: f"{INTENTION_ICONS[x]} {INTENTION_NAMES[x]}"
    )
    
    intent_info = engine.get_intention_description(selected_int)
    st.markdown(f"**{intent_info['name']}**: {intent_info['description']}")
    
    st.markdown("---")
    
    articles = engine.get_articles_by_intention(selected_int, top_n=24)
    if articles:
        cols = st.columns(4)
        for idx, (aid, score) in enumerate(articles):
            with cols[idx % 4]:
                render_product_card(engine, aid, score=score, show_tag=False)

def render_profile(engine, user_id):
    if not user_id: return
    
    st.markdown("<div class='section-title'>Your Fashion DNA</div>", unsafe_allow_html=True)
    
    user_intent = engine.get_user_intention(user_id)
    categories = [INTENTION_NAMES[i] for i in range(10)]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=user_intent.tolist(),
        theta=categories,
        fill='toself',
        fillcolor='rgba(229, 0, 16, 0.2)',
        line=dict(color='#E50010', width=2)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False)),
        showlegend=False,
        margin=dict(l=80, r=80, t=20, b=20),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top Style Affinities")
        top3 = np.argsort(user_intent)[::-1][:3]
        for i, idx in enumerate(top3):
            st.write(f"{i+1}. **{INTENTION_NAMES[idx]}** ({user_intent[idx]:.1%})")
            
    with col2:
        st.markdown("#### Recent History")
        purchases = engine.get_user_purchased_articles(user_id)
        for aid in purchases[-5:]:
            details = engine.get_article_details(aid)
            if details:
                st.caption(f"• {details.get('prod_name')}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    render_header()
    
    try:
        data_path = load_data_from_drive()
        engine = RecommendationEngine(data_path)
        
        with st.sidebar:
            st.markdown(f"<h1 style='color:{COLORS['primary']};'>Member Club</h1>", unsafe_allow_html=True)
            users = engine.get_available_users()
            selected_user = st.selectbox("Sign in as Customer ID:", options=users)
            
            st.markdown("---")
            st.markdown("### 🛍️ Shopping Bag")
            st.caption("Your bag is currently empty.")
            
            st.markdown("---")
            st.markdown("### ℹ️ App Info")
            st.caption("Three-Tower Neural Network")
            st.caption(f"Model AUC: {engine.app_summary.get('model_performance', {}).get('three_tower_auc', 0.8201):.4f}")
            
        tab1, tab2, tab3 = st.tabs(["👗 FOR YOU", "🔍 EXPLORE", "📊 MY STYLE DNA"])
        
        with tab1: render_for_you(engine, selected_user)
        with tab2: render_explore(engine)
        with tab3: render_profile(engine, selected_user)
        
        st.markdown(f"""
            <div style="margin-top: 5rem; padding: 2rem 0; border-top: 1px solid {COLORS['border']}; text-align: center; color: {COLORS['text_muted']}; font-size: 0.8rem;">
                © 2026 H&M Fashion Recommendation System | Research Project
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Something went wrong. Please refresh the page.")
        st.exception(e)

if __name__ == "__main__":
    main()
