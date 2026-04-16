# ============================================================================
# H&M FASHION RECOMMENDATION SYSTEM - STREAMLIT APP
# ============================================================================
# USING GOOGLE DRIVE FOLDERS WITH DIRECT FILE IDs
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
import requests
import gdown

st.set_page_config(
    page_title="H&M Fashion Recommendation",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GOOGLE DRIVE FILE IDs - YOU NEED TO REPLACE THESE WITH YOUR ACTUAL IDs
# ============================================================================
# DATA FOLDER files (replace with your actual file IDs)
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

# IMAGES FOLDER ID - for downloading entire folder
IMAGES_FOLDER_ID = "1cj1f09q4OXcBmG5Hpazn_dYrc9kC7qG6" 

# ============================================================================
# COLOR SCHEME
# ============================================================================
COLORS = {
    'primary': '#E50010', 'secondary': '#000000', 'accent': '#F4F4F4',
    'text': '#333333', 'light': '#FFFFFF', 'success': '#00A651',
    'warning': '#FF6B35', 'info': '#4B86C9'
}

# ============================================================================
# CUSTOM CSS (same as before - keep your existing CSS)
# ============================================================================
st.markdown(f"""
<style>
    /* Your existing CSS here - keep the same */
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
    .product-info {{ padding: 1rem; }}
    .product-name {{
        font-size: 14px;
        font-weight: 600;
        color: {COLORS['text']};
        margin-bottom: 4px;
        line-height: 1.4;
        height: 40px;
        overflow: hidden;
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
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, #ff4757 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }}
    .app-footer {{
        background: {COLORS['secondary']};
        color: white;
        padding: 2rem;
        border-radius: 20px 20px 0 0;
        margin: 3rem -1rem -2rem -1rem;
        text-align: center;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_data_from_drive():
    """Download data files from Google Drive using file IDs"""
    
    progress_container = st.empty()
    
    with progress_container.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📦</div>
                    <h3 style="color: #E50010; margin-bottom: 0.5rem;">Loading Fashion Data</h3>
                    <p class="loading-text" style="color: #666;">Downloading from Google Drive...</p>
                </div>
            """, unsafe_allow_html=True)
            progress_bar = st.progress(0)
    
    try:
        temp_dir = tempfile.mkdtemp()
        data_dir = os.path.join(temp_dir, 'data')
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # Download data files
        progress_bar.progress(10)
        st.text("📥 Downloading data files...")
        
        for i, (filename, file_id) in enumerate(FILE_IDS.items()):
            if file_id and file_id != f'YOUR_{filename.upper().replace(".", "_")}_FILE_ID':
                dest_path = os.path.join(data_dir, filename)
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, dest_path, quiet=True)
                st.text(f"  ✓ {filename}")
            progress_bar.progress(10 + int(i / len(FILE_IDS) * 40))
        
        # Download images folder
        progress_bar.progress(50)
        st.text("📥 Downloading images (this may take a while)...")
        
        import subprocess
        os.chdir(images_dir)
        subprocess.run([
            "gdown", f"https://drive.google.com/drive/folders/{IMAGES_FOLDER_ID}",
            "--folder", "--quiet"
        ], capture_output=True)
        
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_container.empty()
        
        return temp_dir
        
    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Error: {str(e)}")
        raise e

# ============================================================================
# RECOMMENDATION ENGINE (same as before)
# ============================================================================
class RecommendationEngine:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        
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
# UI RENDERING FUNCTIONS
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
        
        selected_user = st.selectbox("Select Customer ID", options=users, index=0)
        
        st.markdown("---")
        st.markdown("### 📊 Profile Overview")
        
        dominant_idx, dominant_score = engine.get_dominant_intention(selected_user)
        confidence = engine.get_user_confidence(selected_user)
        n_purchases = len(engine.get_user_purchased_articles(selected_user))
        intent_info = engine.get_intention_description(dominant_idx)
        
        st.markdown(f"""
            <div style="background: white; border-radius: 12px; padding: 1rem; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
                    <span style="font-size: 12px; color: #888;">ID: {selected_user}</span>
                </div>
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 11px; color: #888;">Dominant Style</div>
                    <div style="font-size: 16px; font-weight: 600;">{intent_info['icon']} {intent_info['name'][:35]}</div>
                    <div style="margin-top: 8px; background: #f0f0f0; border-radius: 10px; height: 6px;">
                        <div style="background: linear-gradient(90deg, #E50010, #ff6b6b); width: {dominant_score*100}%; height: 100%; border-radius: 10px;"></div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; border-top: 1px solid #eee; padding-top: 1rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: 700; color: #E50010;">{n_purchases}</div>
                        <div style="font-size: 10px; color: #888;">Purchases</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: 700; color: #E50010;">{confidence:.0%}</div>
                        <div style="font-size: 10px; color: #888;">Confidence</div>
                    </div>
                </div>
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
            except:
                st.image("https://via.placeholder.com/300x400?text=H&M", use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x400?text=H&M", use_container_width=True)
        
        st.markdown(f"""
            <div class="product-info">
                <div class="product-name">{details.get('prod_name', 'Unknown')[:50]}</div>
                <div class="product-category">{details.get('product_type_name', 'Fashion')}</div>
        """, unsafe_allow_html=True)
        
        if show_intention:
            article_intent = engine.article_intent_dict.get(str(article_id), np.zeros(10))
            top_intent = np.argmax(article_intent)
            intent_info = engine.get_intention_description(top_intent)
            st.markdown(f"<span class='product-intention'>{intent_info['icon']} {intent_info['name'][:20]}</span>", unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)

def render_for_you_tab(engine, user_id):
    if not user_id:
        st.warning("Please select a customer from the sidebar")
        return
    
    dominant_idx, dominant_score = engine.get_dominant_intention(user_id)
    intent_info = engine.get_intention_description(dominant_idx)
    purchased_count = len(engine.get_user_purchased_articles(user_id))
    confidence = engine.get_user_confidence(user_id)
    
    cols = st.columns(4)
    stats = [
        (intent_info['icon'], intent_info['name'][:25], "Style", COLORS['primary']),
        (purchased_count, "Items", "Purchased", COLORS['info']),
        (f"{confidence:.0%}", "Confidence", "Profile", COLORS['success'] if confidence > 0.5 else COLORS['warning']),
        (f"{dominant_score:.0%}", "Match", "Accuracy", COLORS['primary'])
    ]
    
    for col, (value, label, sublabel, color) in zip(cols, stats):
        with col:
            st.markdown(f"""
                <div class="stats-card" style="border-left-color: {color};">
                    <div class="stats-number" style="color: {color};">{value}</div>
                    <div class="stats-label">{label}</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"### ✨ Curated For You - Based on your {intent_info['name'].lower()} preference")
    
    with st.spinner("Finding your perfect matches..."):
        recommendations = engine.recommend_by_intention(user_id, top_n=24)
    
    if recommendations:
        cols = st.columns(4)
        for idx, aid in enumerate(recommendations):
            render_product_card(engine, aid, cols[idx % 4])
    else:
        st.info("No recommendations available")

def render_explore_tab(engine):
    st.markdown("### 🎯 Shop by Intention")
    
    intention_cols = st.columns(5)
    
    for i in range(10):
        with intention_cols[i % 5]:
            intent_info = engine.get_intention_description(i)
            if st.button(f"{intent_info['icon']}\n{intent_info['name'][:20]}", key=f"intent_{i}", use_container_width=True):
                st.session_state.selected_intention = i
                st.rerun()
    
    if st.session_state.get('selected_intention') is not None:
        intent_id = st.session_state.selected_intention
        intent_info = engine.get_intention_description(intent_id)
        
        st.markdown(f"### {intent_info['icon']} {intent_info['name']}")
        st.caption(intent_info['description'])
        
        articles = engine.get_articles_by_intention(intent_id, top_n=24)
        
        if articles:
            cols = st.columns(4)
            for idx, aid in enumerate(articles):
                render_product_card(engine, aid, cols[idx % 4], show_intention=False)
        else:
            st.info("No products found")

def render_style_profile_tab(engine, user_id):
    if not user_id:
        st.warning("Please select a customer from the sidebar")
        return
    
    user_intent = engine.get_user_intention(user_id)
    
    categories = [engine.get_intention_description(i)['name'][:20] for i in range(10)]
    fig = go.Figure(data=go.Scatterpolar(
        r=user_intent.tolist(),
        theta=categories,
        fill='toself',
        fillcolor='rgba(229, 0, 16, 0.2)',
        line=dict(color='#E50010', width=3)
    ))
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🏆 Your Top Styles")
    top3 = np.argsort(user_intent)[::-1][:3]
    for i, idx in enumerate(top3):
        intent = engine.get_intention_description(idx)
        st.markdown(f"{i+1}. **{intent['name']}** - {user_intent[idx]:.1%}")
    
    st.markdown("#### 📜 Recent Purchases")
    purchases = engine.get_user_purchased_articles(user_id)
    st.write(f"Total: {len(purchases)} items")
    for aid in purchases[-5:]:
        article = engine.get_article_details(aid)
        if article:
            st.markdown(f"- {article.get('prod_name', 'Unknown')}")

def render_account_tab(engine, user_id):
    if not user_id:
        st.warning("Please select a customer from the sidebar")
        return
    
    st.markdown("### 👤 My Account")
    st.info("Account management features")
    
    model_auc = engine.app_summary.get('model_performance', {}).get('three_tower_auc', 0.8201)
    st.metric("Three-Tower AUC", f"{model_auc:.4f}")

def render_footer():
    st.markdown("""
        <div class="app-footer">
            <h4>H&M Fashion Recommendation System</h4>
            <p>Powered by Three-Tower Neural Network | AUC: 0.8201 | 10 Intention Categories</p>
        </div>
    """, unsafe_allow_html=True)

def main():
    render_header()
    
    st.warning("""
    ⚠️ **IMPORTANT:** You need to replace the FILE_IDS in the code with your actual Google Drive file IDs.
    
    **How to get file IDs:**
    1. Open each file in Google Drive
    2. Copy the ID from the URL: `https://drive.google.com/file/d/YOUR_FILE_ID/view`
    3. Update the `FILE_IDS` dictionary in the code
    
    **For images folder:** Use the folder ID: `1cj1f09q4OXcBmG5Hpazn_dYrc9kC7qG6`
    """)
    
    st.stop()
    
    # Uncomment below after replacing FILE_IDS
    """
    try:
        data_dir = load_data_from_drive()
        engine = RecommendationEngine(data_dir)
        selected_user = render_sidebar(engine)
        
        tab1, tab2, tab3, tab4 = st.tabs(["FOR YOU", "EXPLORE", "MY STYLE", "ACCOUNT"])
        with tab1: render_for_you_tab(engine, selected_user)
        with tab2: render_explore_tab(engine)
        with tab3: render_style_profile_tab(engine, selected_user)
        with tab4: render_account_tab(engine, selected_user)
        
        render_footer()
    except Exception as e:
        st.error(f"Error: {str(e)}")
    """

if __name__ == "__main__":
    main()
