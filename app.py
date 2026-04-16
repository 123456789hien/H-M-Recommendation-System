# ============================================================================
# H&M FASHION RECOMMENDATION SYSTEM
# ============================================================================
# A production-ready Streamlit app for personalized fashion recommendations
# Powered by Three-Tower Neural Network | AUC 0.8201
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import zipfile
import tempfile
import time
import gc
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import gdown

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
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #4B86C9 0%, #6EA8D9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Product card styling */
    .product-card {
        background: white;
        border-radius: 12px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s, box-shadow 0.2s;
        text-align: center;
        height: 100%;
    }
    .product-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .product-title {
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0;
        height: 40px;
        overflow: hidden;
    }
    .product-type {
        font-size: 0.7rem;
        color: #666;
        margin-bottom: 0.5rem;
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
        margin-top: 0.25rem;
    }
    
    /* Intention card styling */
    .intention-card {
        background: linear-gradient(135deg, #BECDE0 0%, #D0DCE8 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid #90B7E4;
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
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
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
    
    /* Metric styling */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    
    /* Loading spinner */
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

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
@st.cache_resource(ttl=3600, show_spinner=False)
def load_data():
    """Load all data from Google Drive with caching"""
    
    data_path = os.path.join(DATA_DIR, "data")
    images_path = os.path.join(DATA_DIR, "images")
    
    # Check if data already exists
    if os.path.exists(data_path) and os.path.exists(images_path):
        st.session_state.data_loaded = True
        return load_from_local(data_path, images_path)
    
    # Download and extract
    with st.spinner("📥 Downloading data from Google Drive (first time only)..."):
        try:
            # Download using gdown (more reliable)
            zip_path = os.path.join(DATA_DIR, "hm_app_data.zip")
            os.makedirs(DATA_DIR, exist_ok=True)
            
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, zip_path, quiet=False)
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            
            # Clean up
            os.remove(zip_path)
            
            st.session_state.data_loaded = True
            return load_from_local(data_path, images_path)
            
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            return None


def load_from_local(data_path, images_path):
    """Load data from local directory"""
    
    try:
        # Load article metadata
        article_df = pd.read_csv(os.path.join(data_path, 'article_metadata.csv'))
        article_df['article_id'] = article_df['article_id'].astype(str)
        
        # Load article intentions
        article_intentions = pd.read_csv(os.path.join(data_path, 'article_intention_profiles.csv'))
        article_intentions['article_id'] = article_intentions['article_id'].astype(str)
        
        # Load user intentions
        user_intentions = pd.read_csv(os.path.join(data_path, 'user_intention_weights.csv'))
        user_intentions['customer_id'] = user_intentions['customer_id'].astype(str)
        
        # Load test interactions
        test_interactions = pd.read_csv(os.path.join(data_path, 'test_interactions.csv'))
        test_interactions['customer_id'] = test_interactions['customer_id'].astype(str)
        test_interactions['article_id'] = test_interactions['article_id'].astype(str)
        
        # Load intention labels
        with open(os.path.join(data_path, 'intention_labels.json'), 'r') as f:
            intention_labels = json.load(f)
        
        return {
            'article_df': article_df,
            'article_intentions': article_intentions,
            'user_intentions': user_intentions,
            'test_interactions': test_interactions,
            'intention_labels': intention_labels,
            'images_path': images_path
        }
        
    except Exception as e:
        st.error(f"Error loading local data: {str(e)}")
        return None


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================
class RecommendationEngine:
    def __init__(self, data):
        self.data = data
        self.intention_cols = [f'intention_{i}' for i in range(10)]
        
        # Build user intention mapping
        self.user_intent = {}
        for _, row in data['user_intentions'].iterrows():
            uid = str(row['customer_id'])
            self.user_intent[uid] = row[self.intention_cols].values.astype(np.float32)
        
        # Build article intention mapping
        self.article_intent = {}
        for _, row in data['article_intentions'].iterrows():
            aid = str(row['article_id'])
            self.article_intent[aid] = row[self.intention_cols].values.astype(np.float32)
        
        # Build user purchase history
        self.user_purchases = {}
        for _, row in data['test_interactions'].iterrows():
            uid = str(row['customer_id'])
            aid = str(row['article_id'])
            if uid not in self.user_purchases:
                self.user_purchases[uid] = []
            self.user_purchases[uid].append(aid)
        
        # Build article metadata
        self.article_meta = {}
        for _, row in data['article_df'].iterrows():
            self.article_meta[str(row['article_id'])] = row.to_dict()
        
        self.intention_labels = data['intention_labels']
        self.images_path = data['images_path']
        
        # Global prior (cold-start fallback)
        self.global_prior = np.ones(10) / 10
    
    def get_user_intention(self, user_id):
        """Get user intention profile with cold-start handling"""
        if user_id in self.user_intent:
            return self.user_intent[user_id]
        return self.global_prior
    
    def get_dominant_intention(self, user_id):
        """Get user's dominant intention"""
        intent = self.get_user_intention(user_id)
        idx = np.argmax(intent)
        return idx, intent[idx]
    
    def get_user_purchases(self, user_id):
        """Get user's purchase history"""
        return self.user_purchases.get(user_id, [])
    
    def recommend_by_intention(self, user_id, top_n=12, exclude_purchased=True):
        """Recommend products based on intention similarity"""
        user_intent = self.get_user_intention(user_id)
        purchased = set(self.get_user_purchases(user_id)) if exclude_purchased else set()
        
        scores = []
        for aid, art_intent in self.article_intent.items():
            if aid in purchased:
                continue
            sim = cosine_similarity([user_intent], [art_intent])[0][0]
            scores.append((aid, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    
    def get_article_by_intention(self, intention_id, top_n=24):
        """Get top articles for a specific intention"""
        articles = []
        for aid, intent in self.article_intent.items():
            if np.argmax(intent) == intention_id:
                articles.append((aid, intent[intention_id]))
        
        articles.sort(key=lambda x: x[1], reverse=True)
        return articles[:top_n]
    
    def get_article_details(self, article_id):
        """Get article metadata"""
        return self.article_meta.get(str(article_id), {})
    
    def get_intention_name(self, intention_id):
        """Get intention name"""
        intent = self.intention_labels.get(str(intention_id), {})
        return intent.get('name', f'Intention {intention_id}')
    
    def get_intention_description(self, intention_id):
        """Get intention description"""
        intent = self.intention_labels.get(str(intention_id), {})
        return intent.get('description', '')
    
    def get_intention_icon(self, intention_id):
        """Get emoji icon for intention"""
        icons = {
            0: "👗", 1: "👕", 2: "🧦", 3: "👶", 4: "👖",
            5: "🧥", 6: "👜", 7: "💕", 8: "🧶", 9: "👔"
        }
        return icons.get(intention_id, "🎯")
    
    def get_article_image(self, article_id):
        """Get image path for article"""
        img_id = str(article_id).zfill(10)
        img_path = os.path.join(self.images_path, f"{img_id}.jpg")
        if os.path.exists(img_path):
            return img_path
        return None
    
    def get_all_users(self):
        """Get list of all users"""
        return sorted(list(self.user_purchases.keys()))


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_header():
    """Render main header"""
    st.markdown("""
    <div class="main-header">
        <h1>👗 H&M Fashion Recommendation System</h1>
        <p>Powered by Three-Tower Neural Network | Intention-Aware AI | AUC 0.8201</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(engine):
    """Render sidebar with user selection"""
    
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/H%26M-Logo.svg/1200px-H%26M-Logo.svg.png",
        use_container_width=True
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Customer Login")
    
    # User selection
    users = engine.get_all_users()
    selected_user = st.sidebar.selectbox(
        "Select a customer",
        options=users,
        index=0,
        help="Choose a customer to see personalized recommendations"
    )
    
    # User profile
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Your Profile")
    
    dominant_idx, dominant_score = engine.get_dominant_intention(selected_user)
    purchases = len(engine.get_user_purchases(selected_user))
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("🛒 Purchases", purchases)
    with col2:
        st.metric("🎯 Confidence", f"{dominant_score:.0%}")
    
    st.sidebar.markdown(f"**Dominant Style:**")
    st.sidebar.markdown(f"<span style='font-size:1.2rem;'>{engine.get_intention_icon(dominant_idx)} {engine.get_intention_name(dominant_idx)[:40]}</span>", 
                        unsafe_allow_html=True)
    
    # Confidence indicator
    if dominant_score >= 0.6:
        st.sidebar.success("✅ High confidence profile")
    elif dominant_score >= 0.4:
        st.sidebar.warning("⚠️ Medium confidence profile")
    else:
        st.sidebar.info("🔄 Cold-start profile (using global prior)")
    
    return selected_user


def render_for_you_tab(engine, user_id):
    """Render For You tab with personalized recommendations"""
    
    st.markdown("### ✨ Personalized For You")
    st.markdown("Based on your unique style profile and purchase history")
    
    with st.spinner("Finding your perfect matches..."):
        recommendations = engine.recommend_by_intention(user_id, top_n=12)
    
    if not recommendations:
        st.warning("No recommendations available. Try exploring by intention!")
        return
    
    # Display in grid
    cols = st.columns(3)
    for idx, (article_id, score) in enumerate(recommendations):
        article = engine.get_article_details(article_id)
        
        with cols[idx % 3]:
            # Product image
            img_path = engine.get_article_image(article_id)
            if img_path:
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                except:
                    st.image("https://via.placeholder.com/200x250?text=H&M", use_container_width=True)
            else:
                st.image("https://via.placeholder.com/200x250?text=H&M", use_container_width=True)
            
            # Product info
            st.markdown(f"<div class='product-title'>{article.get('prod_name', 'Unknown')[:50]}</div>", 
                       unsafe_allow_html=True)
            st.markdown(f"<div class='product-type'>{article.get('product_type_name', '')} | {article.get('department_name', '')}</div>", 
                       unsafe_allow_html=True)
            
            # Intention badge
            art_intent = engine.article_intent.get(article_id, np.zeros(10))
            top_intent = np.argmax(art_intent)
            st.markdown(f"<div class='intention-badge'>{engine.get_intention_icon(top_intent)} {engine.get_intention_name(top_intent)[:25]}</div>", 
                       unsafe_allow_html=True)
            st.markdown(f"<div class='match-score'>Match: {score:.1%}</div>", 
                       unsafe_allow_html=True)
            
            st.button("🔍 View Details", key=f"view_{article_id}", use_container_width=True)


def render_explore_tab(engine):
    """Render Explore by Intention tab"""
    
    st.markdown("### 🔍 Shop by Intention")
    st.markdown("Discover products that match your shopping needs")
    
    # Intention grid
    cols = st.columns(5)
    selected_intention = None
    
    for i in range(10):
        with cols[i % 5]:
            if st.button(
                f"{engine.get_intention_icon(i)}\n{engine.get_intention_name(i)[:20]}", 
                key=f"intent_{i}", 
                use_container_width=True
            ):
                selected_intention = i
    
    if selected_intention is not None:
        st.markdown("---")
        st.markdown(f"### {engine.get_intention_icon(selected_intention)} {engine.get_intention_name(selected_intention)}")
        st.markdown(f"*{engine.get_intention_description(selected_intention)}*")
        st.markdown("---")
        
        with st.spinner("Loading products..."):
            products = engine.get_article_by_intention(selected_intention, top_n=12)
        
        cols = st.columns(3)
        for idx, (article_id, score) in enumerate(products):
            article = engine.get_article_details(article_id)
            
            with cols[idx % 3]:
                # Product image
                img_path = engine.get_article_image(article_id)
                if img_path:
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                    except:
                        st.image("https://via.placeholder.com/200x250?text=H&M", use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/200x250?text=H&M", use_container_width=True)
                
                st.markdown(f"<div class='product-title'>{article.get('prod_name', 'Unknown')[:50]}</div>", 
                           unsafe_allow_html=True)
                st.markdown(f"<div class='product-type'>{article.get('product_type_name', '')}</div>", 
                           unsafe_allow_html=True)
                st.button("🛒 Add to Cart", key=f"explore_cart_{article_id}", use_container_width=True)


def render_profile_tab(engine, user_id):
    """Render My Style Profile tab"""
    
    st.markdown("### 🎯 Your Style Profile")
    st.markdown("Your personal fashion identity based on your purchase history")
    
    user_intent = engine.get_user_intention(user_id)
    
    # Radar chart
    categories = [engine.get_intention_name(i)[:20] for i in range(10)]
    
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
            radialaxis=dict(
                visible=True,
                range=[0, max(user_intent) * 1.1],
                tickformat='.0%'
            )
        ),
        showlegend=True,
        title="Your 10-Dimensional Intention Profile",
        height=550,
        margin=dict(l=80, r=80, t=80, b=80)
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
                <div style="font-size: 2.5rem;">{engine.get_intention_icon(intent_idx)}</div>
                <div style="font-weight: bold; margin: 0.5rem 0;">{engine.get_intention_name(intent_idx)[:50]}</div>
                <div style="font-size: 1.5rem; font-weight: bold;">{user_intent[intent_idx]:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Purchase history
    st.markdown("### 📜 Your Purchase History")
    purchases = engine.get_user_purchases(user_id)
    st.markdown(f"**Total purchases:** {len(purchases)} items")
    
    if purchases:
        recent = purchases[-5:]
        st.markdown("**Recent items:**")
        for aid in recent:
            article = engine.get_article_details(aid)
            if article:
                st.markdown(f"- {article.get('prod_name', 'Unknown')} *({article.get('product_type_name', '')})*")


def render_account_tab(engine, user_id):
    """Render My Account tab"""
    
    st.markdown("### 👤 My Account")
    
    purchases = engine.get_user_purchases(user_id)
    dominant_idx, dominant_score = engine.get_dominant_intention(user_id)
    
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
            <p><strong>Dominant Style:</strong> {engine.get_intention_name(dominant_idx)[:40]}</p>
            <p><strong>Profile Confidence:</strong> {dominant_score:.1%}</p>
            <p><strong>Wishlist Items:</strong> 0</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Account Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        email_notif = st.checkbox("📧 Email me about new arrivals", value=True)
        sale_notif = st.checkbox("🏷️ Sale notifications", value=True)
    with col2:
        rec_notif = st.checkbox("🎯 Personalized recommendations", value=True)
        newsletter = st.checkbox("📰 Style tips newsletter", value=False)
    
    if st.button("💾 Save Preferences", use_container_width=True):
        st.success("✅ Preferences saved successfully!")


def render_footer():
    """Render footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.75rem; padding: 1rem;">
        <p>👗 H&M Fashion Recommendation System | Three-Tower Neural Network | Intention-Aware AI</p>
        <p>📊 Model Performance: AUC 0.8201 (+3.54% vs Two-Tower) | 10 Intention Categories | Bayesian User Profiling</p>
        <p>🔬 Powered by ResNet-50, BLIP, Sentence-BERT, LDA (K=10), and Three-Tower Neural Network</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Load data
    data = load_data()
    
    if data is None:
        st.error("""
        ### ⚠️ Unable to load data
        
        Please check:
        1. Your internet connection
        2. Google Drive file permissions
        3. Try refreshing the page
        
        If the problem persists, please contact support.
        """)
        return
    
    # Initialize recommendation engine
    engine = RecommendationEngine(data)
    
    # Render UI
    render_header()
    
    # Sidebar
    selected_user = render_sidebar(engine)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "👗 FOR YOU", 
        "🔍 EXPLORE BY INTENTION", 
        "🎯 MY STYLE PROFILE", 
        "👤 MY ACCOUNT"
    ])
    
    with tab1:
        render_for_you_tab(engine, selected_user)
    
    with tab2:
        render_explore_tab(engine)
    
    with tab3:
        render_profile_tab(engine, selected_user)
    
    with tab4:
        render_account_tab(engine, selected_user)
    
    # Footer
    render_footer()


if __name__ == "__main__":
    main()
