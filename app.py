# ============================================================================
# H&M FASHION RECOMMENDATION SYSTEM - STREAMLIT APP
# ============================================================================
# LUẬN ÁN THẠC SĨ - HỆ THỐNG GỢI Ý THỜI TRANG DỰA TRÊN Ý ĐỊNH NGƯỜI DÙNG
# ============================================================================
# CẬP NHẬT: FILE ZIP 600MB - GOOGLE DRIVE ID MỚI
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
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import gdown

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
# CONSTANTS & CONFIG
# ============================================================================
# Google Drive File ID cho file ZIP 600MB
GOOGLE_DRIVE_FILE_ID = "1A3MjmlkiKIYLnDOXFmGJyMgpbpvryoG2"

# Color scheme - H&M Brand Colors
COLORS = {
    'primary': '#E50010',
    'secondary': '#000000',
    'accent': '#F4F4F4',
    'text': '#333333',
    'light': '#FFFFFF',
    'success': '#00A651',
    'warning': '#FF6B35',
    'info': '#4B86C9'
}

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .main .block-container {{
        padding-top: 0;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}
    
    .app-header {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, #ff4757 100%);
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        text-align: center;
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 10px 40px rgba(229, 0, 16, 0.3);
    }}
    
    .app-header h1 {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}
    
    .app-header p {{
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 300;
    }}
    
    .css-1d391kg {{
        background-color: {COLORS['accent']};
    }}
    
    .product-card {{
        background: white;
        border-radius: 16px;
        padding: 0;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        overflow: hidden;
        border: 1px solid #eee;
    }}
    
    .product-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }}
    
    .product-image-container {{
        position: relative;
        overflow: hidden;
        aspect-ratio: 3/4;
    }}
    
    .product-image {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
    }}
    
    .product-card:hover .product-image {{
        transform: scale(1.08);
    }}
    
    .product-badge {{
        position: absolute;
        top: 12px;
        left: 12px;
        background: {COLORS['primary']};
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .product-info {{
        padding: 1rem;
    }}
    
    .product-name {{
        font-size: 14px;
        font-weight: 600;
        color: {COLORS['text']};
        margin-bottom: 4px;
        line-height: 1.4;
        height: 40px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }}
    
    .product-category {{
        font-size: 12px;
        color: #888;
        margin-bottom: 8px;
    }}
    
    .product-intention {{
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 500;
    }}
    
    .intention-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }}
    
    .intention-card {{
        background: white;
        border: 2px solid transparent;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }}
    
    .intention-card:hover {{
        border-color: {COLORS['primary']};
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(229, 0, 16, 0.15);
    }}
    
    .intention-icon {{
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }}
    
    .intention-name {{
        font-size: 14px;
        font-weight: 600;
        color: {COLORS['text']};
        margin-bottom: 0.3rem;
    }}
    
    .intention-desc {{
        font-size: 12px;
        color: #888;
        line-height: 1.4;
    }}
    
    .stats-card {{
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 4px solid {COLORS['primary']};
    }}
    
    .stats-number {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS['primary']};
    }}
    
    .stats-label {{
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .profile-header {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {COLORS['accent']};
        padding: 8px;
        border-radius: 12px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        border-radius: 8px;
        padding: 0 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary']} !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(229, 0, 16, 0.3);
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, #ff4757 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(229, 0, 16, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(229, 0, 16, 0.4);
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    
    .loading-text {{
        animation: pulse 1.5s ease-in-out infinite;
    }}
    
    .app-footer {{
        background: {COLORS['secondary']};
        color: white;
        padding: 2rem;
        border-radius: 20px 20px 0 0;
        margin: 3rem -1rem -2rem -1rem;
        text-align: center;
    }}
    
    @media (max-width: 768px) {{
        .app-header h1 {{
            font-size: 1.8rem;
        }}
        .intention-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}
    
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: #f1f1f1;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['primary']};
        border-radius: 4px;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS - SỬ DỤNG GDOWN
# ============================================================================
@st.cache_resource(show_spinner=False)
def download_and_extract_data():
    """Download và extract dữ liệu từ Google Drive sử dụng gdown"""
    
    progress_container = st.empty()
    
    with progress_container.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📦</div>
                    <h3 style="color: #E50010; margin-bottom: 0.5rem;">Loading Fashion Data</h3>
                    <p class="loading-text" style="color: #666;">Downloading from Google Drive (600MB)...</p>
                    <p style="font-size: 12px; color: #888;">First time may take 3-5 minutes</p>
                </div>
            """, unsafe_allow_html=True)
            progress_bar = st.progress(0)
    
    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        data_dir = os.path.join(temp_dir, 'data')
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        zip_path = os.path.join(temp_dir, "hm_app_data.zip")
        
        progress_bar.progress(10)
        st.text("📥 Downloading 600MB file from Google Drive...")
        
        # Sử dụng gdown để tải file
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, zip_path, quiet=False, fuzzy=True)
        
        # Kiểm tra file
        if not os.path.exists(zip_path):
            raise Exception("Download failed: File not found")
        
        file_size = os.path.getsize(zip_path)
        if file_size == 0:
            raise Exception("Downloaded file is empty")
        
        st.text(f"📦 Downloaded: {file_size / 1024 / 1024:.1f} MB")
        
        progress_bar.progress(50)
        
        # Giải nén
        st.text("📂 Extracting files...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_container.empty()
        
        return temp_dir
        
    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Troubleshooting:\n1. Check your internet connection\n2. Try refreshing the page\n3. File may be too large, consider using a smaller dataset")
        raise e

# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================
class RecommendationEngine:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        
        # Load all data
        with st.spinner("🔄 Initializing recommendation engine..."):
            self.article_df = pd.read_csv(os.path.join(data_dir, 'data', 'article_metadata.csv'))
            self.article_intentions, self.intention_cols = self.load_article_intentions(data_dir)
            self.user_intentions, _ = self.load_user_intentions(data_dir)
            self.intention_labels = self.load_intention_labels(data_dir)
            self.test_interactions = pd.read_csv(os.path.join(data_dir, 'data', 'test_interactions.csv'))
            self.user_confidence_df = self.load_user_confidence(data_dir)
            self.app_summary = self.load_app_summary(data_dir)
            
            self._build_mappings()
    
    def load_article_intentions(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, 'data', 'article_intention_profiles.csv'))
        intention_cols = [f'intention_{i}' for i in range(10)]
        return df, intention_cols
    
    def load_user_intentions(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, 'data', 'user_intention_weights.csv'))
        intention_cols = [f'intention_{i}' for i in range(10)]
        return df, intention_cols
    
    def load_intention_labels(self, data_dir):
        with open(os.path.join(data_dir, 'data', 'intention_labels.json'), 'r') as f:
            return json.load(f)
    
    def load_user_confidence(self, data_dir):
        try:
            return pd.read_csv(os.path.join(data_dir, 'data', 'user_confidence_scores.csv'))
        except:
            return pd.DataFrame()
    
    def load_app_summary(self, data_dir):
        try:
            with open(os.path.join(data_dir, 'data', 'app_summary.json'), 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _build_mappings(self):
        # Build lookup dictionaries
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
        
        article_ids = []
        intentions = []
        for aid, intent in self.article_intent_dict.items():
            if aid not in purchased:
                article_ids.append(aid)
                intentions.append(intent)
        
        if not article_ids:
            return []
        
        similarities = cosine_similarity([user_intent], intentions)[0]
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        return [article_ids[i] for i in top_indices]
    
    def get_article_details(self, article_id):
        return self.article_meta_dict.get(str(article_id))
    
    def get_intention_description(self, intention_id):
        return self.intention_labels.get(str(intention_id), {
            "name": f"Intention {intention_id}",
            "description": "Shopping preference category",
            "icon": "🎯"
        })
    
    def get_dominant_intention(self, user_id):
        user_intent = self.get_user_intention(user_id)
        dominant_idx = np.argmax(user_intent)
        return dominant_idx, user_intent[dominant_idx]
    
    def get_article_image_path(self, article_id):
        img_id = str(article_id).zfill(10)
        
        for root, dirs, files in os.walk(self.images_dir):
            for file in files:
                if file.startswith(img_id) or file == f"{img_id}.jpg":
                    return os.path.join(root, file)
        
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        if os.path.exists(img_path):
            return img_path
            
        return None
    
    def get_available_users(self):
        return sorted(list(self.user_history.keys()))
    
    def get_articles_by_intention(self, intention_id, top_n=24):
        articles = []
        for aid, intent_vec in self.article_intent_dict.items():
            if np.argmax(intent_vec) == intention_id:
                articles.append((aid, intent_vec[intention_id]))
        
        articles.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in articles[:top_n]]

# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_header():
    st.markdown("""
        <div class="app-header">
            <h1>👗 H&M Fashion Recommendation</h1>
            <p>✨ AI-Powered Personal Shopping Experience ✨</p>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar(engine):
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="color: #E50010; font-weight: 700; letter-spacing: -1px;">
                    H&M<span style="font-weight: 300;">AI</span>
                </h2>
                <p style="font-size: 12px; color: #888; margin-top: -10px;">
                    Smart Fashion Recommendations
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 👤 Customer Login")
        
        users = engine.get_available_users()
        if not users:
            st.error("No users available")
            return None
        
        selected_user = st.selectbox(
            "Select Customer ID",
            options=users,
            index=0,
            help="Choose a customer to view personalized recommendations"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Profile Overview")
        
        dominant_idx, dominant_score = engine.get_dominant_intention(selected_user)
        confidence = engine.get_user_confidence(selected_user)
        n_purchases = len(engine.get_user_purchased_articles(selected_user))
        intent_info = engine.get_intention_description(dominant_idx)
        
        if confidence >= 0.6:
            badge_color = COLORS['success']
            badge_text = "High Confidence"
            badge_icon = "✅"
        elif confidence >= 0.4:
            badge_color = COLORS['warning']
            badge_text = "Medium Confidence"
            badge_icon = "⚠️"
        else:
            badge_color = "#999"
            badge_text = "New Customer"
            badge_icon = "🆕"
        
        st.markdown(f"""
            <div style="background: white; border-radius: 12px; padding: 1rem; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <span style="font-size: 12px; color: #888;">ID: {selected_user}</span>
                    <span style="background: {badge_color}20; color: {badge_color}; 
                               padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 600;">
                        {badge_icon} {badge_text}
                    </span>
                </div>
                
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 11px; color: #888; margin-bottom: 4px;">Dominant Style</div>
                    <div style="font-size: 16px; font-weight: 600; color: #333; line-height: 1.3;">
                        {intent_info['icon']} {intent_info['name'][:35]}
                    </div>
                    <div style="margin-top: 8px; background: #f0f0f0; border-radius: 10px; height: 6px;">
                        <div style="background: linear-gradient(90deg, #E50010, #ff6b6b); 
                                    width: {dominant_score*100}%; height: 100%; border-radius: 10px;">
                        </div>
                    </div>
                    <div style="text-align: right; font-size: 11px; color: #666; margin-top: 4px;">
                        {dominant_score:.1%} match
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; 
                            border-top: 1px solid #eee; padding-top: 1rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: 700; color: #E50010;">{n_purchases}</div>
                        <div style="font-size: 10px; color: #888; text-transform: uppercase;">Purchases</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: 700; color: #E50010;">{confidence:.0%}</div>
                        <div style="font-size: 10px; color: #888; text-transform: uppercase;">Confidence</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="background: #f8f9fa; border-radius: 10px; padding: 1rem; margin-top: 1rem;">
                <p style="font-size: 12px; color: #666; margin: 0;">
                    💡 <b>Tip:</b> Use the tabs above to explore recommendations.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        return selected_user

def render_product_card(engine, article_id, col, show_intention=True):
    details = engine.get_article_details(article_id)
    if not details:
        return
    
    img_path = engine.get_article_image_path(article_id)
    
    with col:
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        
        if img_path:
            try:
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
            except Exception:
                st.image("https://via.placeholder.com/300x400?text=H&M", use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x400?text=H&M", use_container_width=True)
        
        st.markdown(f"""
            <div class="product-info">
                <div class="product-name">{details.get('prod_name', 'Unknown Product')}</div>
                <div class="product-category">{details.get('product_type_name', 'Fashion Item')}</div>
        """, unsafe_allow_html=True)
        
        if show_intention:
            article_intent = engine.article_intent_dict.get(str(article_id), np.zeros(10))
            top_intent = np.argmax(article_intent)
            intent_info = engine.get_intention_description(top_intent)
            st.markdown(f"""
                <span class="product-intention">
                    {intent_info['icon']} {intent_info['name'][:20]}
                </span>
            """, unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)

def render_for_you_tab(engine, user_id):
    if not user_id:
        st.warning("👈 Please select a customer from the sidebar")
        return
    
    dominant_idx, dominant_score = engine.get_dominant_intention(user_id)
    intent_info = engine.get_intention_description(dominant_idx)
    purchased_count = len(engine.get_user_purchased_articles(user_id))
    confidence = engine.get_user_confidence(user_id)
    
    cols = st.columns(4)
    stats = [
        (intent_info['icon'], intent_info['name'][:25], "Your Style", COLORS['primary']),
        (purchased_count, "Items Purchased", "Activity", COLORS['info']),
        (f"{confidence:.0%}", "Profile Confidence", "Reliability", COLORS['success'] if confidence > 0.5 else COLORS['warning']),
        (f"{dominant_score:.0%}", "Style Match", "Accuracy", COLORS['primary'])
    ]
    
    for col, (value, label, sublabel, color) in zip(cols, stats):
        with col:
            st.markdown(f"""
                <div class="stats-card" style="border-left-color: {color};">
                    <div class="stats-number" style="color: {color};">{value}</div>
                    <div class="stats-label">{label}</div>
                    <div style="font-size: 11px; color: #aaa; margin-top: 4px;">{sublabel}</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown(f"""
        <h3 style="margin-bottom: 1.5rem;">
            ✨ Curated For You 
            <span style="font-size: 14px; color: #888; font-weight: 400;">
                Based on your {intent_info['name'].lower()} preference
            </span>
        </h3>
    """, unsafe_allow_html=True)
    
    with st.spinner("🎯 Finding your perfect matches..."):
        recommendations = engine.recommend_by_intention(user_id, top_n=24)
    
    if recommendations:
        cols = st.columns(4)
        for idx, aid in enumerate(recommendations):
            render_product_card(engine, aid, cols[idx % 4])
    else:
        st.info("No recommendations available for this user profile.")

def render_explore_tab(engine):
    st.markdown("""
        <h2 style="text-align: center; margin-bottom: 0.5rem;">🎯 Shop by Intention</h2>
        <p style="text-align: center; color: #888; margin-bottom: 2rem;">
            Discover products tailored to your specific shopping needs
        </p>
    """, unsafe_allow_html=True)
    
    intention_cols = st.columns(5)
    
    for i in range(10):
        with intention_cols[i % 5]:
            intent_info = engine.get_intention_description(i)
            if st.button(
                f"{intent_info['icon']}\n\n**{intent_info['name'][:30]}**",
                key=f"intent_btn_{i}",
                use_container_width=True,
                help=intent_info['description']
            ):
                st.session_state.selected_intention = i
                st.rerun()
    
    if st.session_state.get('selected_intention') is not None:
        intent_id = st.session_state.selected_intention
        intent_info = engine.get_intention_description(intent_id)
        
        st.markdown("---")
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1.5rem; border-radius: 16px; margin: 1.5rem 0;">
                <h3 style="margin: 0; color: white;">{intent_info['icon']} {intent_info['name']}</h3>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">{intent_info['description']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        articles = engine.get_articles_by_intention(intent_id, top_n=24)
        
        if articles:
            cols = st.columns(4)
            for idx, aid in enumerate(articles):
                render_product_card(engine, aid, cols[idx % 4], show_intention=False)
        else:
            st.info("No products found for this intention category.")

def render_style_profile_tab(engine, user_id):
    if not user_id:
        st.warning("👈 Please select a customer from the sidebar")
        return
    
    user_intent = engine.get_user_intention(user_id)
    dominant_idx, dominant_score = engine.get_dominant_intention(user_id)
    intent_info = engine.get_intention_description(dominant_idx)
    
    st.markdown(f"""
        <div class="profile-header">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 4rem;">{intent_info['icon']}</div>
                <div>
                    <h2 style="margin: 0; color: white;">Your Style Profile</h2>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">
                        Primary Style: <b>{intent_info['name']}</b> ({dominant_score:.1%})
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        categories = [engine.get_intention_description(i)['name'][:20] for i in range(10)]
        values = user_intent.tolist()
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(229, 0, 16, 0.2)',
            line=dict(color='#E50010', width=3),
            marker=dict(size=8, color='#E50010')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(values) * 1.2], tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=11))
            ),
            showlegend=False,
            height=500,
            margin=dict(l=80, r=80, t=40, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏆 Top Preferences")
        
        top_indices = np.argsort(user_intent)[::-1][:5]
        for rank, idx in enumerate(top_indices, 1):
            intent = engine.get_intention_description(idx)
            percentage = user_intent[idx]
            
            st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 1rem; 
                            background: white; padding: 0.8rem; border-radius: 12px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                    <div style="background: {'#E50010' if rank == 1 else '#ff6b6b' if rank == 2 else '#ffa502' if rank == 3 else '#ddd'}; 
                                color: {'white' if rank <= 3 else '#666'}; 
                                width: 32px; height: 32px; border-radius: 50%; 
                                display: flex; align-items: center; justify-content: center;
                                font-weight: 700; font-size: 14px; margin-right: 12px;">
                        {rank}
                    </div>
                    <div style="flex: 1;">
                        <div style="font-weight: 600; color: #333; font-size: 14px;">
                            {intent['icon']} {intent['name'][:25]}
                        </div>
                        <div style="font-size: 12px; color: #888;">{intent['description'][:40]}...</div>
                    </div>
                    <div style="font-weight: 700; color: #E50010; font-size: 16px;">
                        {percentage:.1%}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📦 Recent Purchase History")
    
    purchased = engine.get_user_purchased_articles(user_id)
    
    if purchased:
        st.markdown(f"<p style='color: #888;'>You've purchased <b>{len(purchased)}</b> items</p>", unsafe_allow_html=True)
        
        recent = purchased[-8:][::-1]
        cols = st.columns(4)
        
        for idx, aid in enumerate(recent):
            details = engine.get_article_details(aid)
            if details:
                with cols[idx % 4]:
                    img_path = engine.get_article_image_path(aid)
                    if img_path:
                        st.image(img_path, use_container_width=True)
                    st.caption(f"**{details.get('prod_name', 'Unknown')[:30]}**")
    else:
        st.info("No purchase history available for this customer.")

def render_account_tab(engine, user_id):
    if not user_id:
        st.warning("👈 Please select a customer from the sidebar")
        return
    
    st.markdown('<h2 style="text-align: center; margin-bottom: 2rem;">👤 Account Settings</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style="background: white; border-radius: 16px; padding: 1.5rem; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                <h4 style="margin-top: 0; color: #333;">📋 Account Information</h4>
        """, unsafe_allow_html=True)
        
        purchased = len(engine.get_user_purchased_articles(user_id))
        confidence = engine.get_user_confidence(user_id)
        
        st.markdown(f"""
            <div style="margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; padding: 0.8rem 0; border-bottom: 1px solid #eee;">
                    <span style="color: #888;">Customer ID</span>
                    <span style="font-weight: 600; font-family: monospace;">{user_id}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.8rem 0; border-bottom: 1px solid #eee;">
                    <span style="color: #888;">Account Type</span>
                    <span style="font-weight: 600;">Premium Member</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.8rem 0; border-bottom: 1px solid #eee;">
                    <span style="color: #888;">Total Purchases</span>
                    <span style="font-weight: 600; color: #E50010;">{purchased} items</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.8rem 0;">
                    <span style="color: #888;">Profile Confidence</span>
                    <span style="font-weight: 600; color: #E50010;">{confidence:.0%}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background: white; border-radius: 16px; padding: 1.5rem; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                <h4 style="margin-top: 0; color: #333;">⚙️ Preferences</h4>
        """, unsafe_allow_html=True)
        
        st.checkbox("📧 Email notifications for new arrivals", value=True)
        st.checkbox("🎯 Personalized recommendations", value=True)
        st.checkbox("🏷️ Sale and discount alerts", value=True)
        st.checkbox("📊 Weekly style reports", value=False)
        
        if st.button("💾 Save Preferences", use_container_width=True):
            st.success("✅ Preferences saved successfully!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    model_auc = engine.app_summary.get('model_performance', {}).get('three_tower_auc', 0.8201)
    model_improvement = engine.app_summary.get('model_performance', {}).get('improvement', '+3.54%')
    
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                    border-radius: 16px; padding: 1.5rem; margin-top: 2rem;">
            <h4 style="margin-top: 0; color: #333;">🎓 About This System</h4>
            <p style="color: #666; line-height: 1.6; margin-bottom: 1rem;">
                This recommendation engine is powered by a <b>Three-Tower Neural Network</b> 
                with intention-aware architecture, developed for Master's thesis research.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="font-size: 24px; font-weight: 700; color: #E50010;">{model_auc:.4f}</div>
                    <div style="font-size: 12px; color: #888;">AUC Score</div>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="font-size: 24px; font-weight: 700; color: #E50010;">{model_improvement}</div>
                    <div style="font-size: 12px; color: #888;">vs Two-Tower</div>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="font-size: 24px; font-weight: 700; color: #E50010;">10</div>
                    <div style="font-size: 12px; color: #888;">Intention Categories</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown("""
        <div class="app-footer">
            <h4 style="margin: 0 0 1rem 0; color: white;">H&M Fashion Recommendation System</h4>
            <p style="margin: 0; opacity: 0.8; font-size: 14px;">
                Powered by Three-Tower Neural Network | Intention-Aware Recommendations
            </p>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.6; font-size: 12px;">
                Master's Thesis Research Project | AUC: 0.8201 | 10 Intention Categories
            </p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    render_header()
    
    try:
        with st.spinner(""):
            data_dir = download_and_extract_data()
            engine = RecommendationEngine(data_dir)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return
    
    selected_user = render_sidebar(engine)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "👗 **FOR YOU**", "🔍 **EXPLORE**", "🎯 **MY STYLE**", "👤 **ACCOUNT**"
    ])
    
    with tab1:
        render_for_you_tab(engine, selected_user)
    with tab2:
        render_explore_tab(engine)
    with tab3:
        render_style_profile_tab(engine, selected_user)
    with tab4:
        render_account_tab(engine, selected_user)
    
    render_footer()

if __name__ == "__main__":
    main()
