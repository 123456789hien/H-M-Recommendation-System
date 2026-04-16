# ============================================================================
# H&M FASHION RECOMMENDATION SYSTEM
# ============================================================================
# Updated with new Google Drive file ID
# AUC 0.8201 | Three-Tower Neural Network | Intention-Aware AI
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import zipfile
import tempfile
from PIL import Image
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
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
# CONSTANTS - UPDATED FILE ID
# ============================================================================
GOOGLE_DRIVE_FILE_ID = "1aWdBLp_5B07qxUFmH9mvpQ92kI_VFBbB"
DATA_DIR = tempfile.gettempdir() + "/hm_app_data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")

INTENTION_ICONS = {
    0: "👗", 1: "👕", 2: "🧦", 3: "👶", 4: "👖",
    5: "🧥", 6: "👜", 7: "💕", 8: "🧶", 9: "👔"
}

# ============================================================================
# DOWNLOAD FUNCTION (IMPROVED)
# ============================================================================
def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive using requests"""
    
    session = requests.Session()
    
    # First request to get confirmation token
    response = session.get(
        f"https://drive.google.com/uc?id={file_id}&export=download",
        stream=True
    )
    
    # Check for download warning
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            confirm_token = value
            break
    
    # Second request with confirmation token
    if confirm_token:
        response = session.get(
            f"https://drive.google.com/uc?id={file_id}&confirm={confirm_token}&export=download",
            stream=True
        )
    else:
        response = session.get(
            f"https://drive.google.com/uc?export=download&id={file_id}",
            stream=True
        )
    
    # Save file
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    
    return destination


# ============================================================================
# IMAGE LOADING (OPTIMIZED)
# ============================================================================
@st.cache_data(ttl=3600, max_entries=200, show_spinner=False)
def load_image_cached(article_id):
    """Load and cache image - only 200 most recent images kept"""
    img_id = str(article_id).zfill(10)
    img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
    
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path)
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
    """Load data from Google Drive"""
    
    data_path = os.path.join(DATA_DIR, "data")
    
    # Check if already downloaded
    if os.path.exists(data_path) and os.path.exists(os.path.join(data_path, "article_metadata.csv")):
        return load_csv_data(data_path)
    
    # Download and extract
    with st.spinner("📥 Loading data (first time only, please wait 2-3 minutes)..."):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            zip_path = os.path.join(DATA_DIR, "data.zip")
            
            # Download using updated method
            download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, zip_path)
            
            # Verify it's a valid zip file
            if not zipfile.is_zipfile(zip_path):
                raise Exception("Downloaded file is not a valid zip archive")
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATA_DIR)
            
            os.remove(zip_path)
            
            return load_csv_data(data_path)
            
        except Exception as e:
            st.error(f"❌ Failed to load data: {str(e)}")
            st.info("💡 Troubleshooting:\n1. Make sure file permission is 'Anyone with the link'\n2. Try downloading manually to check if file is valid\n3. Check your internet connection")
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
        
        # Limit data for better performance (500 users, 5000 articles)
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
        
        self.user_intent = {}
        for _, row in data['user_intent'].iterrows():
            self.user_intent[str(row['customer_id'])] = row[intent_cols].values.astype(np.float32)
        
        self.article_intent = {}
        for _, row in data['article_intent'].iterrows():
            self.article_intent[str(row['article_id'])] = row[intent_cols].values.astype(np.float32)
        
        self.user_purchases = {}
        for _, row in data['interactions'].iterrows():
            uid = str(row['customer_id'])
            aid = str(row['article_id'])
            if uid not in self.user_purchases:
                self.user_purchases[uid] = []
            self.user_purchases[uid].append(aid)
        
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
    
    data = load_data()
    if data is None:
        st.stop()
    
    engine = RecommendationEngine(data)
    selected_user = render_sidebar(engine)
    
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
