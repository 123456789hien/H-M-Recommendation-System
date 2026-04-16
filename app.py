import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import zipfile
import requests
from io import BytesIO
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import time
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="H&M Fashion Recommendation",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS & ASSETS
# ============================================================================
FILE_ID = "1-wRrYq1f5R8XAMoVJHB8S_dt8MbfilcH"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# Custom CSS for a professional e-commerce look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #000000 0%, #333333 100%);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }
    
    /* Product Card Styling */
    .product-card {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border: 1px solid #eee;
        transition: all 0.3s ease;
        text-align: center;
        height: 100%;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.08);
        border-color: #000;
    }
    
    .product-title {
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 0.8rem;
        color: #333;
        height: 2.4rem;
        overflow: hidden;
    }
    
    .product-price {
        font-weight: 700;
        color: #e50019; /* H&M Red */
        margin-top: 0.5rem;
    }
    
    .product-type {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Intention Badge */
    .intention-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        background-color: #f0f0f0;
        color: #555;
        margin-top: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-around;
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 2rem;
    }
    
    .metric-card {
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #000;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eee;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #000 !important;
        color: #000 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING ENGINE
# ============================================================================
@st.cache_resource
def download_and_extract_data():
    """Download ZIP from Google Drive and extract to temp directory"""
    try:
        response = requests.get(DOWNLOAD_URL, stream=True)
        # Handle Google Drive confirmation page
        if "download_warning" in response.text:
            import re
            confirm_token = re.search(r'confirm=([^&]+)', response.text)
            if confirm_token:
                confirm_url = f"https://drive.google.com/uc?export=download&id={FILE_ID}&confirm={confirm_token.group(1)}"
                response = requests.get(confirm_url, stream=True)
        
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "hm_app_data.zip")
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_dir)
        
        return extract_dir
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

class RecommendationEngine:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        
        # Load all CSV/JSON data
        self.article_df = pd.read_csv(os.path.join(data_dir, 'data', 'article_metadata.csv'))
        self.article_intentions = pd.read_csv(os.path.join(data_dir, 'data', 'article_intention_profiles.csv'))
        self.user_intentions = pd.read_csv(os.path.join(data_dir, 'data', 'user_intention_weights.csv'))
        self.test_interactions = pd.read_csv(os.path.join(data_dir, 'data', 'test_interactions.csv'))
        
        with open(os.path.join(data_dir, 'data', 'intention_labels.json'), 'r') as f:
            self.intention_labels = json.load(f)
            
        self.intention_cols = [f'intention_{i}' for i in range(10)]
        self._build_mappings()
    
    def _build_mappings(self):
        # Mappings for fast lookup
        self.user_intent_dict = {str(row['customer_id']): row[self.intention_cols].values.astype(np.float32) 
                                for _, row in self.user_intentions.iterrows()}
        
        self.article_intent_dict = {str(row['article_id']): row[self.intention_cols].values.astype(np.float32) 
                                   for _, row in self.article_intentions.iterrows()}
        
        self.user_history = {}
        for _, row in self.test_interactions.iterrows():
            uid, aid = str(row['customer_id']), str(row['article_id'])
            self.user_history.setdefault(uid, []).append(aid)
            
        self.article_meta_dict = {str(row['article_id']): row.to_dict() for _, row in self.article_df.iterrows()}
    
    def get_user_intention(self, user_id):
        return self.user_intent_dict.get(user_id, np.ones(10) / 10)
    
    def recommend(self, user_id, top_n=12):
        user_intent = self.get_user_intention(user_id)
        scores = []
        purchased = set(self.user_history.get(user_id, []))
        
        for aid, art_intent in self.article_intent_dict.items():
            if aid in purchased: continue
            sim = cosine_similarity([user_intent], [art_intent])[0][0]
            scores.append((aid, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scores[:top_n]]

    def get_article_image(self, article_id):
        img_name = f"{str(article_id).zfill(10)}.jpg"
        img_path = os.path.join(self.images_dir, img_name)
        if os.path.exists(img_path):
            return Image.open(img_path)
        return None

# ============================================================================
# UI COMPONENTS
# ============================================================================
def product_card(engine, article_id):
    details = engine.article_meta_dict.get(str(article_id))
    if not details: return
    
    img = engine.get_article_image(article_id)
    
    with st.container():
        st.markdown(f"""
        <div class="product-card">
            <div class="product-type">{details.get('product_group_name', 'Fashion')}</div>
            <div class="product-title">{details.get('prod_name', 'H&M Item')}</div>
            <div class="product-price">${np.random.uniform(19.99, 59.99):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        if img:
            st.image(img, use_container_width=True)
        else:
            st.image("https://via.placeholder.com/200x300?text=No+Image", use_container_width=True)
        
        if st.button("View Details", key=f"btn_{article_id}", use_container_width=True):
            st.session_state.selected_item = article_id

def render_for_you(engine, user_id):
    st.subheader("✨ Recommended for You")
    st.write("Based on your unique fashion intention profile.")
    
    recs = engine.recommend(user_id)
    cols = st.columns(4)
    for i, aid in enumerate(recs):
        with cols[i % 4]:
            product_card(engine, aid)

def render_style_profile(engine, user_id):
    st.subheader("🎯 Your Fashion DNA")
    user_intent = engine.get_user_intention(user_id)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Radar Chart
        categories = [engine.intention_labels[str(i)]['name'] for i in range(10)]
        fig = go.Figure(data=go.Scatterpolar(
            r=user_intent,
            theta=categories,
            fill='toself',
            marker=dict(color='#000000'),
            line=dict(color='#000000', width=2)
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(user_intent)*1.2])),
            showlegend=False,
            height=450,
            margin=dict(l=80, r=80, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### Top Style Drivers")
        top_idx = np.argsort(user_intent)[::-1][:3]
        for idx in top_idx:
            intent_info = engine.intention_labels[str(idx)]
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 5px solid #000; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <h4 style="margin:0;">{intent_info['name']}</h4>
                <p style="font-size: 0.85rem; color: #666; margin: 0.5rem 0;">{intent_info['description']}</p>
                <div style="font-weight: bold; color: #000;">Match Score: {user_intent[idx]:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

def render_explore(engine):
    st.subheader("🔍 Explore by Collection")
    selected_intent = st.selectbox("Select a Fashion Intention", 
                                  options=range(10), 
                                  format_func=lambda x: engine.intention_labels[str(x)]['name'])
    
    st.info(f"💡 {engine.intention_labels[str(selected_intent)]['description']}")
    
    # Find items with high score in this intention
    art_scores = []
    for aid, intents in engine.article_intent_dict.items():
        art_scores.append((aid, intents[selected_intent]))
    
    art_scores.sort(key=lambda x: x[1], reverse=True)
    top_items = [aid for aid, _ in art_scores[:12]]
    
    cols = st.columns(4)
    for i, aid in enumerate(top_items):
        with cols[i % 4]:
            product_card(engine, aid)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>H&M FASHION RECOMMENDATION</h1>
        <p>Intention-Aware Neural Discovery Engine • Master's Thesis Project</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Engine
    with st.spinner("Initializing AI Engine..."):
        data_dir = download_and_extract_data()
        if not data_dir: return
        engine = RecommendationEngine(data_dir)

    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/H%26M-Logo.svg/1200px-H%26M-Logo.svg.png", width=100)
        st.markdown("---")
        st.subheader("User Selection")
        test_users = pd.read_csv(os.path.join(data_dir, 'data', 'sampled_user_ids.csv'))['customer_id'].tolist()
        selected_user = st.selectbox("Select Customer ID", test_users)
        
        st.markdown("---")
        st.markdown("### Model Stats")
        st.metric("Model AUC", "0.8201", "+3.54%")
        st.metric("Intentions", "10 Categories")
        
        st.markdown("---")
        st.caption("© 2026 H&M Recommendation Thesis")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🛍️ FOR YOU", "🔍 EXPLORE", "🎯 STYLE PROFILE"])
    
    with tab1:
        render_for_you(engine, selected_user)
        
    with tab2:
        render_explore(engine)
        
    with tab3:
        render_style_profile(engine, selected_user)

    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #888; font-size: 0.8rem;'>Built with Streamlit • Data Source: H&M Personalized Fashion Recommendations Kaggle</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
