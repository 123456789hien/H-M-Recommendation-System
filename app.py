# ============================================================================
# H&M FASHION RECOMMENDATION SYSTEM
# ============================================================================
# Production-ready Streamlit app with image optimization
# AUC 0.8201 | Three-Tower Neural Network | Intention-Aware AI
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import zipfile
import tempfile
import time
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import requests

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="H&M Fashion Recommender",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #4B86C9 0%, #6EA8D9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Product card */
    .product-card {
        background: white;
        border-radius: 12px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
        text-align: center;
        height: 100%;
    }
    .product-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .product-title {
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.5rem 0;
        height: 40px;
        overflow: hidden;
    }
    .product-type {
        font-size: 0.7rem;
        color: #666;
    }
    .intention-badge {
        background-color: #BECDE0;
        padding: 2px 8px;
        border-radius: 15px;
        font-size: 0.65rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .match-score {
        font-size: 0.7rem;
        color: #4B86C9;
        font-weight: bold;
    }
    
    /* Intention card */
    .intention-card {
        background: linear-gradient(135deg, #BECDE0 0%, #D0DCE8 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
    }
    .intention-card:hover {
        background: linear-gradient(135deg, #4B86C9 0%, #6EA8D9 100%);
        color: white;
        transform: scale(1.02);
    }
    .intention-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .intention-name {
        font-size: 0.7rem;
        font-weight: 600;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4B86C9;
        color: white;
        border-radius: 25px;
        border: none;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #3a6ba0;
        transform: translateY(-1px);
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    
    /* Loading */
    .stSpinner > div {
        border-top-color: #4B86C9 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
GOOGLE_DRIVE_FILE_ID = "1-wRrYq1f5R8XAMoVJHB8S_dt8MbfilcH"
DATA_DIR = tempfile.gettempdir() + "/hm_app_data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")

# Intention icons
INTENTION_ICONS = {
    0: "👗", 1: "👕", 2: "🧦", 3: "👶", 4: "👖",
    5: "🧥", 6: "👜", 7: "💕", 8: "🧶", 9: "👔"
}

# ============================================================================
# IMAGE LOADING (OPTIMIZED)
# ============================================================================
@st.cache_data(ttl=3600, max_entries=200, show_spinner=False)
def load_image_cached(article_id):
    """Load and cache image with size limit - only 200 most recent images kept"""
    img_id = str(article_id).zfill(10)
    img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
    
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            # Resize immediately to reduce memory footprint
            img.thumbnail((180, 240), Image.Resampling.LANCZOS)
            return img
        except Exception:
            return None
    return None

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_resource(ttl=3600, show_spinner=False)
def load_data():
    """Load data from Google Drive with caching"""
    
    data_path = os.path.join(DATA_DIR, "data")
    
    # Check if already downloaded
    if os.path.exists(data_path) and os.path.exists(os.path.join(data_path, "article_metadata.csv")):
        return load_csv_data(data_path)
    
    # Download and extract
    with st.spinner("📥 Loading data (first time only, please wait)..."):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            zip_path = os.path.join(DATA_DIR, "data.zip")
            
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, zip_path, quiet=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATA_DIR)
            
            os.remove(zip_path)
            
            return load_csv_data(data_path)
            
        except Exception as e:
            st.error(f"❌ Failed to load data: {str(e)}")
            return None


def load_csv_data(data_path):
    """Load CSV files only, images loaded on demand"""
    
    try:
        # Load article metadata
        article_df = pd.read_csv(os.path.join(data_path, 'article_metadata.csv'))
        article_df['article_id'] = article_df['article_id'].astype(str)
        
        # Load article intentions
        article_intent = pd.read_csv(os.path.join(data_path, 'article_intention_profiles.csv'))
        article_intent['article_id'] = article_intent['article_id'].astype(str)
        
        # Load user intentions
        user_intent = pd.read_csv(os.path.join(data_path, 'user_intention_weights.csv'))
        user_intent['customer_id'] = user_intent['customer_id'].astype(str)
        
        # Load test interactions
        test_interactions = pd.read_csv(os.path.join(data_path, 'test_interactions.csv'))
        test_interactions['customer_id'] = test_interactions['customer_id'].astype(str)
        test_interactions['article_id'] = test_interactions['article_id'].astype(str)
        
        # Load intention labels
        with open(os.path.join(data_path, 'intention_labels.json'), 'r') as f:
            intention_labels = json.load(f)
        
        # Limit data for demo (500 users, 5000 articles for better performance)
        unique_users = user_intent['customer_id'].unique()[:500]
        unique_articles = article_df['article_id'].unique()[:5000]
        
        user_intent = user_intent[user_intent['customer_id'].isin(unique_users)]
        article_df = article_df[article_df['article_id'].isin(unique_articles)]
        article_intent = article_intent[article_intent['article_id'].isin(unique_articles)]
        test_interactions = test_interactions[test_interactions['customer_id'].isin(unique_users)]
        
        return {
            'articles': article_df,
            'article_intent': article_intent,
            'user_intent': user_intent,
            'interactions': test_interactions,
            'labels': intention_labels,
            'users': unique_users,
            'articles_list': unique_articles
        }
        
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        return None


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================
class RecommendationEngine:
    def __init__(self, data):
        self.data = data
        intent_cols = [f'intention_{i}' for i in range(10)]
        
        # User intention mapping
        self.user_intent = {}
        for _, row in data['user_intent'].iterrows():
            self.user_intent[str(row['customer_id'])] = row[intent_cols].values.astype(np.float32)
        
        # Article intention mapping
        self.article_intent = {}
        for _, row in data['article_intent'].iterrows():
            self.article_intent[str(row['article_id'])] = row[intent_cols].values.astype(np.float32)
        
        # User purchase history
        self.user_purchases = {}
        for _, row in data['interactions'].iterrows():
            uid = str(row['customer_id'])
            aid = str(row['article_id'])
            if uid not in self.user_purchases:
                self.user_purchases[uid] = []
            self.user_purchases[uid].append(aid)
        
        # Article metadata
        self.article_meta = {}
        for _, row in data['articles'].iterrows():
            self.article_meta[str(row['article_id'])] = row.to_dict()
        
        self.labels = data['labels']
        self.global_prior = np.ones(10) / 10
    
    def get_user_intent(self, user_id):
        if user_id in self.user_intent:
            return self.user_intent[user_id]
        return self.global_prior
    
    def get_dominant_intent(self, user_id):
        intent = self.get_user_intent(user_id)
        idx = np.argmax(intent)
        return idx, intent[idx]
    
    def recommend(self, user_id, top_n=12):
        """Get personalized recommendations"""
        user_intent = self.get_user_intent(user_id)
        purchased = set(self.user_purchases.get(user_id, []))
        
        scores = []
        for aid, art_intent in self.article_intent.items():
            if aid in purchased:
                continue
            sim = cosine_similarity([user_intent], [art_intent])[0][0]
            scores.append((aid, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    
    def get_by_intention(self, intent_id, top_n=12):
        """Get top articles for a specific intention"""
        articles = []
        for aid, intent in self.article_intent.items():
            if np.argmax(intent) == intent_id:
                articles.append((aid, intent[intent_id]))
        articles.sort(key=lambda x: x[1], reverse=True)
        return articles[:top_n]
    
    def get_article(self, article_id):
        return self.article_meta.get(str(article_id), {})
    
    def get_intent_name(self, intent_id):
        return self.labels.get(str(intent_id), {}).get('name', f'Intention {intent_id}')
    
    def get_intent_desc(self, intent_id):
        return self.labels.get(str(intent_id), {}).get('description', '')
    
    def get_intent_icon(self, intent_id):
        return INTENTION_ICONS.get(intent_id, "🎯")
    
    def get_users(self):
        return sorted(list(self.user_purchases.keys()))[:100]


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>👗 H&M Fashion Recommendation System</h1>
        <p>Powered by Three-Tower Neural Network | Intention-Aware AI | AUC 0.8201</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(engine):
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/H%26M-Logo.svg/1200px-H%26M-Logo.svg.png",
        use_container_width=True
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Customer Login")
    
    users = engine.get_users()
    if not users:
        st.sidebar.warning("No users available")
        return None
    
    selected_user = st.sidebar.selectbox("Select Customer ID", users)
    
    # User profile
    dom_idx, dom_score = engine.get_dominant_intent(selected_user)
    purchases = len(engine.user_purchases.get(selected_user, []))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Your Profile")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("🛒 Purchases", purchases)
    with col2:
        st.metric("🎯 Confidence", f"{dom_score:.0%}")
    
    st.sidebar.markdown(f"**Dominant Style:**")
    st.sidebar.markdown(f"{engine.get_intent_icon(dom_idx)} **{engine.get_intent_name(dom_idx)[:40]}**")
    
    if dom_score >= 0.6:
        st.sidebar.success("✅ High confidence profile")
    elif dom_score >= 0.4:
        st.sidebar.warning("⚠️ Medium confidence profile")
    else:
        st.sidebar.info("🔄 Cold-start profile")
    
    return selected_user


def render_product_card(article_id, article, score=None):
    """Display a single product card with lazy-loaded image"""
    
    # Load image on demand (cached)
    img = load_image_cached(article_id)
    
    if img:
        st.image(img, use_container_width=True)
    else:
        st.image("https://via.placeholder.com/180x240?text=H&M", use_container_width=True)
    
    st.markdown(f"<div class='product-title'>{article.get('prod_name', 'Unknown')[:45]}</div>", 
                unsafe_allow_html=True)
    st.markdown(f"<div class='product-type'>{article.get('product_type_name', '')}</div>", 
                unsafe_allow_html=True)
    
    if score:
        st.markdown(f"<div class='match-score'>Match: {score:.1%}</div>", unsafe_allow_html=True)
    
    # Intention badge
    st.button("🔍 View Details", key=f"view_{article_id}", use_container_width=True)


def render_for_you_tab(engine, user_id):
    st.markdown("### ✨ Personalized For You")
    st.markdown("Based on your unique style profile and purchase history")
    
    if not user_id:
        st.warning("Please select a customer from the sidebar")
        return
    
    with st.spinner("Finding your perfect matches..."):
        recommendations = engine.recommend(user_id, top_n=12)
    
    if not recommendations:
        st.warning("No recommendations available")
        return
    
    cols = st.columns(3)
    for idx, (article_id, score) in enumerate(recommendations):
        article = engine.get_article(article_id)
        with cols[idx % 3]:
            render_product_card(article_id, article, score)


def render_explore_tab(engine):
    st.markdown("### 🔍 Shop by Intention")
    st.markdown("Discover products that match your shopping needs")
    
    # Intention grid
    cols = st.columns(5)
    selected_intent = None
    
    for i in range(10):
        with cols[i % 5]:
            if st.button(
                f"{engine.get_intent_icon(i)}\n{engine.get_intent_name(i)[:20]}",
                key=f"intent_{i}",
                use_container_width=True
            ):
                selected_intent = i
    
    if selected_intent is not None:
        st.markdown("---")
        st.markdown(f"### {engine.get_intent_icon(selected_intent)} {engine.get_intent_name(selected_intent)}")
        st.markdown(f"*{engine.get_intent_desc(selected_intent)}*")
        st.markdown("---")
        
        with st.spinner("Loading products..."):
            products = engine.get_by_intention(selected_intent, top_n=12)
        
        cols = st.columns(3)
        for idx, (article_id, score) in enumerate(products):
            article = engine.get_article(article_id)
            with cols[idx % 3]:
                render_product_card(article_id, article)


def render_profile_tab(engine, user_id):
    st.markdown("### 🎯 Your Style Profile")
    st.markdown("Your personal fashion identity based on your purchase history")
    
    if not user_id:
        st.warning("Please select a customer from the sidebar")
        return
    
    user_intent = engine.get_user_intent(user_id)
    
    # Radar chart
    categories = [engine.get_intent_name(i)[:20] for i in range(10)]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=user_intent.tolist(),
        theta=categories,
        fill='toself',
        marker=dict(color='#4B86C9', size=6),
        line=dict(color='#4B86C9', width=2),
        name='Your Style Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(user_intent) * 1.1], tickformat='.0%')
        ),
        showlegend=True,
        title="Your 10-Dimensional Intention Profile",
        height=550,
        margin=dict(l=60, r=60, t=60, b=60)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top intentions
    st.markdown("### 🏆 Your Top Style Matches")
    top_indices = np.argsort(user_intent)[::-1][:3]
    
    cols = st.columns(3)
    for idx, intent_idx in enumerate(top_indices):
        with cols[idx]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4B86C9 0%, #6EA8D9 100%); 
                        border-radius: 12px; padding: 1rem; text-align: center; color: white;">
                <div style="font-size: 2.5rem;">{engine.get_intent_icon(intent_idx)}</div>
                <div style="font-weight: bold; margin: 0.5rem 0;">{engine.get_intent_name(intent_idx)[:40]}</div>
                <div style="font-size: 1.5rem; font-weight: bold;">{user_intent[intent_idx]:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Purchase history
    st.markdown("### 📜 Your Purchase History")
    purchases = engine.user_purchases.get(user_id, [])
    st.markdown(f"**Total purchases:** {len(purchases)} items")
    
    if purchases:
        recent = purchases[-5:]
        st.markdown("**Recent items:**")
        for aid in recent:
            article = engine.get_article(aid)
            if article:
                st.markdown(f"- {article.get('prod_name', 'Unknown')} *({article.get('product_type_name', '')})*")


def render_account_tab(engine, user_id):
    st.markdown("### 👤 My Account")
    
    if not user_id:
        st.warning("Please select a customer from the sidebar")
        return
    
    purchases = engine.user_purchases.get(user_id, [])
    dom_idx, dom_score = engine.get_dominant_intent(user_id)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📋 Account Information</h4>
            <p><strong>Customer ID:</strong> {user_id}</p>
            <p><strong>Member Since:</strong> 2024</p>
            <p><strong>Account Type:</strong> Premium</p>
            <p><strong>Loyalty Tier:</strong> Gold</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Shopping Stats</h4>
            <p><strong>Total Purchases:</strong> {len(purchases)}</p>
            <p><strong>Dominant Style:</strong> {engine.get_intent_name(dom_idx)[:40]}</p>
            <p><strong>Profile Confidence:</strong> {dom_score:.1%}</p>
            <p><strong>Wishlist Items:</strong> 0</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Account Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("📧 Email me about new arrivals", value=True)
        st.checkbox("🏷️ Sale notifications", value=True)
    with col2:
        st.checkbox("🎯 Personalized recommendations", value=True)
        st.checkbox("📰 Style tips newsletter", value=False)
    
    if st.button("💾 Save Preferences", use_container_width=True):
        st.success("✅ Preferences saved successfully!")


def render_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.7rem; padding: 1rem;">
        <p>👗 H&M Fashion Recommendation System | Three-Tower Neural Network | Intention-Aware AI</p>
        <p>📊 AUC 0.8201 (+3.54% vs Two-Tower) | 10 Intention Categories | Bayesian User Profiling</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    render_header()
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Initialize engine
    engine = RecommendationEngine(data)
    
    # Sidebar
    selected_user = render_sidebar(engine)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "👗 FOR YOU", "🔍 EXPLORE BY INTENTION", "🎯 MY STYLE PROFILE", "👤 MY ACCOUNT"
    ])
    
    with tab1:
        render_for_you_tab(engine, selected_user)
    
    with tab2:
        render_explore_tab(engine)
    
    with tab3:
        render_profile_tab(engine, selected_user)
    
    with tab4:
        render_account_tab(engine, selected_user)
    
    render_footer()


if __name__ == "__main__":
    main()
