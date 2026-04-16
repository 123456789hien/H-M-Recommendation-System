import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import zipfile
import gdown
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import warnings
import shutil

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

# Custom CSS for a professional e-commerce look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #000000 0%, #333333 100%);
        padding: 2.5rem; border-radius: 15px; text-align: center; color: white;
        margin-bottom: 2rem; box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .product-card {
        background-color: white; border-radius: 12px; padding: 1rem;
        margin-bottom: 1.5rem; border: 1px solid #eee; transition: all 0.3s ease;
        text-align: center; height: 100%;
    }
    .product-card:hover {
        transform: translateY(-5px); box-shadow: 0 12px 20px rgba(0,0,0,0.08); border-color: #000;
    }
    .product-title { font-weight: 600; font-size: 0.9rem; margin-top: 0.8rem; color: #333; height: 2.4rem; overflow: hidden; }
    .product-price { font-weight: 700; color: #e50019; margin-top: 0.5rem; }
    .product-type { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING ENGINE (OPTIMIZED)
# ============================================================================
@st.cache_resource
def get_data_engine(file_id):
    """Memory-efficient data loading and engine initialization"""
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "data.zip")
    url = f'https://drive.google.com/uc?id={file_id}'
    
    try:
        # Download
        gdown.download(url, zip_path, quiet=False)
        
        # Open ZIP without extracting all files to save disk/RAM
        zf = zipfile.ZipFile(zip_path, 'r')
        
        # Detect base path inside ZIP
        all_files = zf.namelist()
        base_path = ""
        # Look for the data directory
        for f in all_files:
            if "data/article_metadata.csv" in f:
                base_path = f.replace("data/article_metadata.csv", "")
                break
        
        # Load CSVs directly from ZIP
        def read_zip_csv(filename):
            full_p = os.path.join(base_path, filename)
            with zf.open(full_p) as f:
                return pd.read_csv(f)

        article_df = read_zip_csv('data/article_metadata.csv')
        article_intentions = read_zip_csv('data/article_intention_profiles.csv')
        user_intentions = read_zip_csv('data/user_intention_weights.csv')
        test_interactions = read_zip_csv('data/test_interactions.csv')
        
        with zf.open(os.path.join(base_path, 'data/intention_labels.json')) as f:
            intention_labels = json.load(f)
            
        # Initialize Engine
        engine = RecommendationEngine(zf, base_path, article_df, article_intentions, user_intentions, test_interactions, intention_labels)
        return engine
    except Exception as e:
        st.error(f"Critical Error: {e}")
        return None

class RecommendationEngine:
    def __init__(self, zip_file, base_path, article_df, article_intentions, user_intentions, test_interactions, intention_labels):
        self.zf = zip_file
        self.base_path = base_path
        self.article_df = article_df
        self.article_intentions = article_intentions
        self.user_intentions = user_intentions
        self.test_interactions = test_interactions
        self.intention_labels = intention_labels
        
        self.intention_cols = [f'intention_{i}' for i in range(10)]
        self._build_mappings()
    
    def _build_mappings(self):
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
        img_name = f"images/{str(article_id).zfill(10)}.jpg"
        full_path = os.path.join(self.base_path, img_name)
        try:
            with self.zf.open(full_path) as f:
                return Image.open(f).copy()
        except:
            return None
            
    def get_intention_info(self, intent_id):
        """Safely get intention name and description"""
        # Ensure ID is a string for JSON lookup
        id_str = str(intent_id)
        info = self.intention_labels.get(id_str, {})
        # Fallback values if fields are missing
        return {
            "name": info.get("name", f"Intention {id_str}"),
            "description": info.get("description", "No description available for this fashion intention.")
        }

# ============================================================================
# UI COMPONENTS
# ============================================================================
def product_card(engine, article_id):
    details = engine.article_meta_dict.get(str(article_id))
    if not details: return
    img = engine.get_article_image(article_id)
    with st.container():
        st.markdown(f"""<div class="product-card">
            <div class="product-type">{details.get('product_group_name', 'Fashion')}</div>
            <div class="product-title">{details.get('prod_name', 'H&M Item')}</div>
            <div class="product-price">${np.random.uniform(19.99, 59.99):.2f}</div>
        </div>""", unsafe_allow_html=True)
        if img: st.image(img, use_container_width=True)
        else: st.image("https://via.placeholder.com/200x300?text=No+Image", use_container_width=True)
        st.button("View Details", key=f"btn_{article_id}", use_container_width=True)

def render_for_you(engine, user_id):
    st.subheader("✨ Recommended for You")
    recs = engine.recommend(user_id)
    cols = st.columns(4)
    for i, aid in enumerate(recs):
        with cols[i % 4]: product_card(engine, aid)

def render_style_profile(engine, user_id):
    st.subheader("🎯 Your Fashion DNA")
    user_intent = engine.get_user_intention(user_id)
    col1, col2 = st.columns([1, 1])
    with col1:
        # Get labels safely
        categories = [engine.get_intention_info(i)['name'] for i in range(10)]
        fig = go.Figure(data=go.Scatterpolar(r=user_intent, theta=categories, fill='toself', marker=dict(color='#000000'), line=dict(color='#000000', width=2)))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(user_intent)*1.2])), showlegend=False, height=450, margin=dict(l=80, r=80, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### Top Style Drivers")
        top_idx = np.argsort(user_intent)[::-1][:3]
        for idx in top_idx:
            info = engine.get_intention_info(idx)
            st.markdown(f"""<div style="background: white; padding: 1rem; border-radius: 10px; border-left: 5px solid #000; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <h4 style="margin:0;">{info['name']}</h4>
                <p style="font-size: 0.85rem; color: #666; margin: 0.5rem 0;">{info['description']}</p>
                <div style="font-weight: bold; color: #000;">Match Score: {user_intent[idx]:.1%}</div>
            </div>""", unsafe_allow_html=True)

def render_explore(engine):
    st.subheader("🔍 Explore by Collection")
    selected_intent = st.selectbox("Select a Fashion Intention", options=range(10), 
                                  format_func=lambda x: engine.get_intention_info(x)['name'])
    
    info = engine.get_intention_info(selected_intent)
    st.info(f"💡 {info['description']}")
    
    art_scores = [(aid, intents[selected_intent]) for aid, intents in engine.article_intent_dict.items()]
    art_scores.sort(key=lambda x: x[1], reverse=True)
    cols = st.columns(4)
    for i, (aid, _) in enumerate(art_scores[:12]):
        with cols[i % 4]: product_card(engine, aid)

def main():
    st.markdown("""<div class="main-header"><h1>H&M FASHION RECOMMENDATION</h1><p>Intention-Aware Neural Discovery Engine • Master's Thesis Project</p></div>""", unsafe_allow_html=True)
    with st.spinner("Initializing AI Engine..."):
        engine = get_data_engine(FILE_ID)
        if not engine: return
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/H%26M-Logo.svg/1200px-H%26M-Logo.svg.png", width=100)
        st.markdown("---")
        st.subheader("User Selection")
        try:
            # Safely check for sampled_user_ids.csv
            sampled_path = os.path.join(engine.base_path, 'data/sampled_user_ids.csv')
            with engine.zf.open(sampled_path) as f:
                test_users = pd.read_csv(f)['customer_id'].tolist()
        except:
            test_users = list(engine.user_intent_dict.keys())[:100]
        selected_user = st.selectbox("Select Customer ID", test_users)
        st.markdown("---")
        st.markdown("### Model Stats")
        st.metric("Model AUC", "0.8201", "+3.54%")
        st.metric("Intentions", "10 Categories")
        st.caption("© 2026 H&M Recommendation Thesis")
    tab1, tab2, tab3 = st.tabs(["🛍️ FOR YOU", "🔍 EXPLORE", "🎯 STYLE PROFILE"])
    with tab1: render_for_you(engine, selected_user)
    with tab2: render_explore(engine)
    with tab3: render_style_profile(engine, selected_user)
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #888; font-size: 0.8rem;'>Built with Streamlit • Data Source: H&M Personalized Fashion Recommendations Kaggle</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
