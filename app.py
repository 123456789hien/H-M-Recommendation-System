# ============================================================================
# H&M ENTERPRISE BUSINESS INTELLIGENCE APP
# ============================================================================
# Strategic BI Dashboard | Three-Tower vs Two-Tower Comparison
# Data from: https://drive.google.com/drive/folders/1-gPW3AAVJOns0PeaR-qna5z1L7Wh6nlD
# Images from: https://drive.google.com/drive/folders/1cj1f09q4OXcBmG5Hpazn_dYrc9kC7qG6
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
import time
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
import requests
import subprocess

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="H&M Strategic BI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================
# Google Drive folder IDs
DATA_FOLDER_ID = "1-gPW3AAVJOns0PeaR-qna5z1L7Wh6nlD"
IMAGES_FOLDER_ID = "1cj1f09q4OXcBmG5Hpazn_dYrc9kC7qG6"

# File IDs from data folder (REAL DATA)
FILE_IDS = {
    'article_metadata.csv': '1RjZmAdpGvQCQHeKpEL30dlTyRenWU1GY',
    'article_intention_profiles.csv': '1aHDWsO8tA2dtKd7bNkk85gk9DP9mNx9M',
    'user_intention_weights.csv': '1C0J3k0FLxCOCxtbLL_dzJ1TDmxWw8rv9',
    'test_interactions.csv': '1AmaZ6DOqTxOOCkpCeRerHz1AyibVoYuG',
    'sampled_user_ids.csv': '1wxbgGcs7K-cUUC8Xm9xEgHyPmXqwE-7w',
    'intention_labels.json': '1Xsw0wM2Wvyo_Mi4PUqfpUYqdDyOEU4bH',
    'user_confidence_scores.csv': '1sa6t6Oun06YpMoJSz7YwN4lufdGYuW6o',
    'customers_cleaned.csv': '1fXH8bSUorehRkbMT2_ROUJUzvBPKzHCO',
    'app_summary.json': '1JJN21tQ4uQ89q-wNvQ1wfnwqV0r7qbHN',
    'intention_summary.csv': '1VJFmGp-RnH4n8N7cwaYIn3DXfGfBcLlL',
    'user_dominant_intention_dist.csv': '1bXqgS02sUDRnPYGgn-2zs0aR3jchhXnj'
}

# 10 INTENTIONS (from your intention_labels.json)
INTENTION_NAMES = {
    0: "Ladieswear Full Body: Special Occasion Dressing",
    1: "Ladieswear Upper Body: Everyday Workwear Comfort",
    2: "Unisex Dark Basics: Utilitarian Necessity Purchase",
    3: "Baby Full Body: Infant & Nurturing Care",
    4: "Unisex Lower Body: Functional Versatility Seeking",
    5: "Children's Upper Body: Trendy & Casual Provisioning",
    6: "Ladies Accessories & Footwear: Hedonic Purchase",
    7: "Ladieswear Underwear: Intimate Self-Care",
    8: "Ladieswear Knitwear: Premium Quality Investment",
    9: "Menswear Shirts: Professional Identity Expression"
}

INTENTION_ICONS = {
    0: "👗", 1: "👕", 2: "🧦", 3: "👶", 4: "👖",
    5: "🧥", 6: "👜", 7: "💕", 8: "🧶", 9: "👔"
}

INTENTION_COLORS = {
    0: "#E67E22", 1: "#2ECC71", 2: "#2C2C2C", 3: "#4ECDC4", 4: "#1B6CA8",
    5: "#27AE60", 6: "#9B59B6", 7: "#E74C3C", 8: "#F39C12", 9: "#1E5496"
}

COLORS = {
    'primary': '#2E86C1',
    'secondary': '#1A5276',
    'success': '#27AE60',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'gold': '#F1C40F',
    'dark': '#2C3E50',
    'light': '#ECF0F1',
    'white': '#FFFFFF'
}

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * {{ font-family: 'Inter', sans-serif; }}
    
    .main-header {{
        background: linear-gradient(135deg, #1A5276 0%, #2E86C1 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }}
    
    .main-header h1 {{ font-size: 1.8rem; font-weight: 700; margin: 0; }}
    .main-header p {{ font-size: 0.9rem; opacity: 0.85; margin: 0.2rem 0 0 0; }}
    
    .kpi-card {{
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #2E86C1;
        transition: all 0.2s;
        height: 100%;
    }}
    
    .kpi-card:hover {{ transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.1); }}
    .kpi-number {{ font-size: 2rem; font-weight: 700; color: #2C3E50; }}
    .kpi-label {{ font-size: 0.75rem; color: #7F8C8D; text-transform: uppercase; letter-spacing: 0.5px; }}
    
    .product-card {{
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
        border: 1px solid #eee;
        height: 100%;
    }}
    
    .product-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        border-color: #2E86C1;
    }}
    
    .product-image {{
        width: 100%;
        aspect-ratio: 3/4;
        background: #f8f9fa;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }}
    
    .product-image img {{ width: 100%; height: 100%; object-fit: cover; }}
    .product-info {{ padding: 0.8rem; }}
    .product-name {{ font-size: 0.85rem; font-weight: 500; color: #2C3E50; height: 2.6rem; overflow: hidden; }}
    
    .intention-badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 600;
        color: white;
        margin-top: 0.3rem;
    }}
    
    .footer {{
        text-align: center;
        padding: 2rem 0;
        color: #95A5A6;
        font-size: 0.8rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }}
    
    .model-comparison-card {{
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        text-align: center;
        border-top: 4px solid #2E86C1;
        height: 100%;
    }}
    
    .model-comparison-card .value {{ font-size: 2.5rem; font-weight: 700; color: #2C3E50; }}
    .model-comparison-card .label {{ font-size: 0.8rem; color: #7F8C8D; text-transform: uppercase; }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DOWNLOAD FUNCTION - FIXED FOR GOOGLE DRIVE
# ============================================================================
def download_file_from_gdrive(file_id, destination):
    """
    Download file from Google Drive using requests.
    Handles the confirmation page that gdown fails to bypass.
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Check for confirmation token
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(url, stream=True)
            break
    
    # Check if we got HTML instead of file
    content_type = response.headers.get('content-type', '')
    if 'text/html' in content_type and 'download_warning' not in response.text:
        # Try alternative URL
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0"
        response = session.get(url, stream=True)
    
    # Save file
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    
    return destination

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_data():
    """Download and extract all data from Google Drive"""
    progress_container = st.empty()
    
    with progress_container.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">📊</div>
                    <h3 style="color: #2C3E50;">Loading Strategic Data</h3>
                    <p style="color: #7F8C8D;">Downloading from Google Drive...</p>
                </div>
            """, unsafe_allow_html=True)
            progress_bar = st.progress(0)
    
    try:
        temp_dir = tempfile.mkdtemp()
        data_dir = os.path.join(temp_dir, 'data')
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # Download data files using fixed function
        for i, (filename, file_id) in enumerate(FILE_IDS.items()):
            dest_path = os.path.join(data_dir, filename)
            download_file_from_gdrive(file_id, dest_path)
            st.text(f"  ✓ {filename}")
            progress_bar.progress(10 + int(i / len(FILE_IDS) * 40))
        
        # Download images folder
        progress_bar.progress(50)
        st.text("📥 Downloading images...")
        
        os.chdir(images_dir)
        folder_url = f"https://drive.google.com/drive/folders/{IMAGES_FOLDER_ID}"
        subprocess.run(["gdown", folder_url, "--folder", "--quiet"], capture_output=True)
        
        # Count images
        image_count = 0
        for root, dirs, files in os.walk(images_dir):
            image_count += len([f for f in files if f.endswith('.jpg')])
        st.text(f"  ✓ Downloaded {image_count} images")
        
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_container.empty()
        return temp_dir
        
    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Data loading failed: {str(e)}")
        return None

# ============================================================================
# BI ENGINE
# ============================================================================
# ============================================================================
# BI ENGINE - COMPLETE FIX
# ============================================================================
class BIEngine:
    def __init__(self, data_dir):
        """
        Initialize the BI Engine with all data from the data directory.
        
        Args:
            data_dir: Path to the temporary directory containing data and images
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        
        # Load all data files
        self._load_data_files(data_dir)
        
        # Define intention columns
        self.intention_cols = [f'intention_{i}' for i in range(10)]
        
        # Build mappings for fast lookup
        self._build_mappings()
    
    def _load_data_files(self, data_dir):
        """Load all data files with error handling and fallbacks"""
        try:
            self.article_df = pd.read_csv(os.path.join(data_dir, 'data', 'article_metadata.csv'))
        except FileNotFoundError:
            st.error("❌ article_metadata.csv not found")
            self.article_df = pd.DataFrame()
        
        try:
            self.article_intentions = pd.read_csv(os.path.join(data_dir, 'data', 'article_intention_profiles.csv'))
        except FileNotFoundError:
            st.error("❌ article_intention_profiles.csv not found")
            self.article_intentions = pd.DataFrame()
        
        try:
            self.user_intentions = pd.read_csv(os.path.join(data_dir, 'data', 'user_intention_weights.csv'))
        except FileNotFoundError:
            st.error("❌ user_intention_weights.csv not found")
            self.user_intentions = pd.DataFrame()
        
        try:
            self.test_interactions = pd.read_csv(os.path.join(data_dir, 'data', 'test_interactions.csv'))
        except FileNotFoundError:
            st.warning("⚠️ test_interactions.csv not found")
            self.test_interactions = pd.DataFrame()
        
        # Load JSON files
        try:
            with open(os.path.join(data_dir, 'data', 'intention_labels.json'), 'r') as f:
                self.intention_labels = json.load(f)
        except FileNotFoundError:
            st.warning("⚠️ intention_labels.json not found. Using default labels.")
            self.intention_labels = {}
        
        try:
            with open(os.path.join(data_dir, 'data', 'app_summary.json'), 'r') as f:
                self.app_summary = json.load(f)
        except FileNotFoundError:
            st.warning("⚠️ app_summary.json not found. Using default values.")
            self.app_summary = {'model_performance': {}}
        
        # Load CSV files with fallbacks
        try:
            self.intention_summary = pd.read_csv(os.path.join(data_dir, 'data', 'intention_summary.csv'))
        except FileNotFoundError:
            st.warning("⚠️ intention_summary.csv not found. Will generate from data.")
            self.intention_summary = pd.DataFrame()
        
        try:
            self.user_confidence = pd.read_csv(os.path.join(data_dir, 'data', 'user_confidence_scores.csv'))
        except FileNotFoundError:
            st.warning("⚠️ user_confidence_scores.csv not found. Will use defaults.")
            self.user_confidence = pd.DataFrame()
        
        # Load dominant distribution with fallback
        self.dominant_dist = self._load_dominant_distribution(data_dir)
    
    def _load_dominant_distribution(self, data_dir):
        """
        Load user dominant intention distribution from file.
        If file is missing or empty, generate from user_intention_weights.
        """
        file_path = os.path.join(data_dir, 'data', 'user_dominant_intention_dist.csv')
        
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                st.warning("⚠️ user_dominant_intention_dist.csv is empty. Generating from user intention weights...")
                return self._create_dominant_distribution_from_weights()
            
            # Validate required columns exist
            if 'dominant_intention' not in df.columns or 'user_count' not in df.columns:
                st.warning("⚠️ user_dominant_intention_dist.csv has wrong columns. Regenerating...")
                return self._create_dominant_distribution_from_weights()
            
            return df
            
        except FileNotFoundError:
            st.warning("⚠️ user_dominant_intention_dist.csv not found. Generating from user intention weights...")
            return self._create_dominant_distribution_from_weights()
        except Exception as e:
            st.warning(f"⚠️ Error loading dominant distribution: {str(e)}. Using fallback...")
            return self._create_dominant_distribution_from_weights()
    
    def _create_dominant_distribution_from_weights(self):
        """
        Create dominant intention distribution from user_intention_weights.
        This calculates the dominant intention for each user and counts them.
        """
        if self.user_intentions.empty:
            st.warning("⚠️ No user intention data available. Using empty distribution.")
            return pd.DataFrame(columns=['dominant_intention', 'user_count'])
        
        dist = {}
        for _, row in self.user_intentions.iterrows():
            try:
                weights = row[self.intention_cols].values.astype(np.float32)
                dominant = np.argmax(weights)
                dist[dominant] = dist.get(dominant, 0) + 1
            except (KeyError, ValueError) as e:
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {'dominant_intention': k, 'user_count': v}
            for k, v in dist.items()
        ])
        
        st.text(f"✅ Generated dominant distribution with {len(df)} intentions")
        return df
    
    def _build_mappings(self):
        """Build dictionary mappings for fast lookup of articles and users"""
        # Article intention mappings
        self.article_intent_dict = {}
        if not self.article_intentions.empty:
            for _, row in self.article_intentions.iterrows():
                try:
                    article_id = str(row['article_id'])
                    self.article_intent_dict[article_id] = row[self.intention_cols].values.astype(np.float32)
                except (KeyError, ValueError):
                    continue
        
        # User intention mappings  
        self.user_intent_dict = {}
        if not self.user_intentions.empty:
            for _, row in self.user_intentions.iterrows():
                try:
                    customer_id = str(row['customer_id'])
                    self.user_intent_dict[customer_id] = row[self.intention_cols].values.astype(np.float32)
                except (KeyError, ValueError):
                    continue
        
        # Article metadata mappings
        self.article_meta_dict = {}
        if not self.article_df.empty:
            for _, row in self.article_df.iterrows():
                try:
                    article_id = str(row['article_id'])
                    self.article_meta_dict[article_id] = row.to_dict()
                except (KeyError):
                    continue
    
    # ========================================================================
    # GETTER METHODS
    # ========================================================================
    
    def get_article_details(self, article_id):
        """
        Get detailed information for a specific article.
        
        Args:
            article_id: Article ID (int or str)
            
        Returns:
            dict: Article details or empty dict if not found
        """
        article_id = str(article_id)
        return self.article_meta_dict.get(article_id, {})
    
    def get_image_path(self, article_id):
        """
        Get the local image path for a specific article.
        
        Args:
            article_id: Article ID (int or str)
            
        Returns:
            str: Path to image file or None if not found
        """
        img_id = str(article_id).zfill(10)
        for root, dirs, files in os.walk(self.images_dir):
            for file in files:
                if file == f"{img_id}.jpg" or file.startswith(img_id):
                    return os.path.join(root, file)
        return None
    
    def get_intention_name(self, i):
        """Get the name of an intention"""
        key = str(i)
        if key in self.intention_labels:
            return self.intention_labels[key].get('name', INTENTION_NAMES.get(i, f'Intention {i}'))
        return INTENTION_NAMES.get(i, f'Intention {i}')
    
    def get_intention_icon(self, i):
        """Get the icon for an intention"""
        return INTENTION_ICONS.get(i, '🎯')
    
    def get_intention_color(self, i):
        """Get the color for an intention"""
        return INTENTION_COLORS.get(i, '#95A5A6')
    
    def get_intention_price_tier(self, i):
        """Get the price tier for an intention"""
        key = str(i)
        if key in self.intention_labels:
            return self.intention_labels[key].get('price_tier', 'N/A')
        return 'N/A'
    
    def get_intention_mean_price(self, i):
        """Get the mean price for an intention"""
        key = str(i)
        if key in self.intention_labels:
            return self.intention_labels[key].get('mean_price', 0)
        return 0
    
    def get_intention_article_count(self, i):
        """Get the article count for an intention"""
        key = str(i)
        if key in self.intention_labels:
            return self.intention_labels[key].get('article_count', 0)
        return 0
    
    def get_intention_article_share(self, i):
        """Get the article share percentage for an intention"""
        key = str(i)
        if key in self.intention_labels:
            return self.intention_labels[key].get('article_share', 0)
        return 0
    
    def get_model_performance(self):
        """Get model performance metrics"""
        perf = self.app_summary.get('model_performance', {})
        return {
            'three_tower_auc': perf.get('three_tower_auc', 0.8201),
            'two_tower_auc': perf.get('two_tower_auc', 0.7921),
            'improvement': perf.get('improvement', '+3.54%')
        }
    
    def get_user_distribution(self):
        """
        Get user distribution across intentions.
        
        Returns:
            dict: {intention_id: user_count}
        """
        dist = {}
        
        if self.dominant_dist.empty:
            # Generate fallback distribution
            return self._create_fallback_distribution()
        
        for _, row in self.dominant_dist.iterrows():
            try:
                intent = int(row['dominant_intention'])
                count = int(row['user_count'])
                dist[intent] = count
            except (KeyError, ValueError) as e:
                continue
        
        # If no data was loaded, use fallback
        if sum(dist.values()) == 0:
            return self._create_fallback_distribution()
        
        return dist
    
    def _create_fallback_distribution(self):
        """
        Create fallback distribution from user_intention_weights.
        Used when dominant_dist is empty or invalid.
        """
        if self.user_intentions.empty:
            # Return equal distribution if no user data
            return {i: 1 for i in range(10)}
        
        dist = {}
        for _, row in self.user_intentions.iterrows():
            try:
                weights = row[self.intention_cols].values.astype(np.float32)
                dominant = np.argmax(weights)
                dist[dominant] = dist.get(dominant, 0) + 1
            except (KeyError, ValueError):
                continue
        
        return dist
    
    def get_article_distribution(self):
        """
        Get article distribution across intentions.
        
        Returns:
            dict: {intention_id: article_count}
        """
        dist = {}
        if self.article_intentions.empty:
            return {i: 0 for i in range(10)}
        
        for _, row in self.article_intentions.iterrows():
            try:
                weights = row[self.intention_cols].values.astype(np.float32)
                intent = np.argmax(weights)
                dist[intent] = dist.get(intent, 0) + 1
            except (KeyError, ValueError):
                continue
        
        return dist
    
    def get_supply_demand_gap(self):
        """
        Calculate the supply-demand gap for each intention.
        
        Returns:
            dict: {intention_id: gap_percentage_points}
        """
        supply = self.get_article_distribution()
        demand = self.get_user_distribution()
        
        total_supply = sum(supply.values())
        total_demand = sum(demand.values())
        
        gaps = {}
        for i in range(10):
            # Calculate percentages
            sup_pct = supply.get(i, 0) / total_supply * 100 if total_supply > 0 else 0
            dem_pct = demand.get(i, 0) / total_demand * 100 if total_demand > 0 else 0
            
            # Gap = Demand % - Supply % (positive means under-supplied)
            gaps[i] = dem_pct - sup_pct
        
        return gaps
    
    def get_intent_summary(self):
        """
        Get comprehensive summary for all intentions.
        
        Returns:
            pd.DataFrame: Complete intention summary with metrics and recommendations
        """
        supply = self.get_article_distribution()
        demand = self.get_user_distribution()
        gaps = self.get_supply_demand_gap()
        
        total_supply = sum(supply.values())
        total_demand = sum(demand.values())
        
        summary = []
        for i in range(10):
            sup = supply.get(i, 0)
            dem = demand.get(i, 0)
            gap = gaps.get(i, 0)
            
            # Determine strategy based on gap
            if gap > 3:
                strategy = "EXPAND"
                strategy_color = "#27AE60"
                strategy_icon = "⬆️"
            elif gap < -3:
                strategy = "RATIONALIZE"
                strategy_color = "#E74C3C"
                strategy_icon = "⬇️"
            else:
                strategy = "MAINTAIN"
                strategy_color = "#F39C12"
                strategy_icon = "➡️"
            
            summary.append({
                'id': i,
                'intention_id': f'T{i}',
                'name': self.get_intention_name(i),
                'icon': self.get_intention_icon(i),
                'color': self.get_intention_color(i),
                'supply': sup,
                'supply_pct': sup / total_supply * 100 if total_supply > 0 else 0,
                'demand': dem,
                'demand_pct': dem / total_demand * 100 if total_demand > 0 else 0,
                'gap': gap,
                'strategy': strategy,
                'strategy_color': strategy_color,
                'strategy_icon': strategy_icon,
                'price_tier': self.get_intention_price_tier(i),
                'mean_price': self.get_intention_mean_price(i),
                'article_count': self.get_intention_article_count(i),
                'article_share': self.get_intention_article_share(i)
            })
        
        return pd.DataFrame(summary)

# ============================================================================
# RENDER FUNCTIONS
# ============================================================================
def render_model_comparison(engine):
    perf = engine.get_model_performance()
    
    st.markdown("### 🧠 Model Performance: Three-Tower vs Two-Tower")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="model-comparison-card" style="border-top-color: #2E86C1;">
                <div class="label">Three-Tower</div>
                <div class="value">{perf['three_tower_auc']:.4f}</div>
                <div style="font-size: 0.8rem; color: #27AE60;">AUC Score</div>
                <div style="margin-top: 0.5rem; background: #2E86C1; color: white; padding: 0.3rem; border-radius: 8px; font-weight: 600;">
                    Main Model
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="model-comparison-card" style="border-top-color: #95A5A6;">
                <div class="label">Two-Tower Baseline</div>
                <div class="value">{perf['two_tower_auc']:.4f}</div>
                <div style="font-size: 0.8rem; color: #7F8C8D;">AUC Score</div>
                <div style="margin-top: 0.5rem; background: #95A5A6; color: white; padding: 0.3rem; border-radius: 8px; font-weight: 600;">
                    Baseline
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="model-comparison-card" style="border-top-color: #27AE60;">
                <div class="label">Improvement</div>
                <div class="value" style="color: #27AE60;">{perf['improvement']}</div>
                <div style="font-size: 0.8rem; color: #27AE60;">AUC Gain</div>
                <div style="margin-top: 0.5rem; background: #27AE60; color: white; padding: 0.3rem; border-radius: 8px; font-weight: 600;">
                    t = 103.12, p &lt; 0.001
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Ablation
    st.markdown("#### 🔬 Ablation Study: Tower Contributions")
    
    ablation_data = {
        'Tower': ['Tower 1 (Visual)', 'Tower 2 (Semantic+Demo)', 'Tower 3 (Intention)', 'Interaction Effects'],
        'Contribution (%)': [37.9, 18.2, 34.1, 9.8],
        'Color': ['#3498DB', '#2ECC71', '#E74C3C', '#95A5A6']
    }
    ablation_df = pd.DataFrame(ablation_data)
    
    fig = px.pie(
        ablation_df,
        values='Contribution (%)',
        names='Tower',
        color='Tower',
        color_discrete_sequence=ablation_df['Color'],
        title='Contribution to AUC Gain (Total +3.54%)',
        height=350
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("""
        💡 **Key Finding:** Tower 3 (Intention Alignment) contributes 34.1% of total AUC gain 
        with only **0.2% of model parameters**.
    """)

def render_supply_demand_gap(engine, chart_key="sd_gap_main"):
    st.markdown("### 📊 Supply-Demand Gap Analysis")
    
    summary = engine.get_intent_summary()
    summary = summary.sort_values('gap', ascending=False)
    
    fig = go.Figure()
    colors = ['#27AE60' if x > 0 else '#E74C3C' if x < 0 else '#F39C12' for x in summary['gap']]
    
    fig.add_trace(go.Bar(
        x=summary['name'],
        y=summary['gap'],
        marker_color=colors,
        text=[f"{g:+.1f}pp" for g in summary['gap']],
        textposition='outside',
        customdata=summary[['supply_pct', 'demand_pct']].values
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=3, line_dash="dot", line_color="#27AE60", annotation_text="Expand Threshold (+3pp)")
    fig.add_hline(y=-3, line_dash="dot", line_color="#E74C3C", annotation_text="Rationalize Threshold (-3pp)")
    
    fig.update_layout(
        title='Supply-Demand Gap by Intention',
        height=450,
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        yaxis_title='Gap (Demand - Supply) percentage points'
    )

    st.plotly_chart(fig, use_container_width=True, key=f"plotly_{chart_key}")

def render_intention_stats(engine):
    st.markdown("### 📊 Intention Statistics")
    
    summary = engine.get_intent_summary()
    
    display_df = summary[['icon', 'intention_id', 'name', 'price_tier', 'mean_price', 
                          'article_count', 'article_share', 'demand_pct', 'supply_pct', 'gap', 'strategy']].copy()
    display_df.columns = ['', 'ID', 'Intention', 'Price Tier', 'Mean Price', 
                          'Articles', 'Article %', 'Demand %', 'Supply %', 'Gap (pp)', 'Strategy']
    display_df['Mean Price'] = display_df['Mean Price'].apply(lambda x: f"${x:.4f}")
    display_df['Article %'] = display_df['Article %'].apply(lambda x: f"{x*100:.2f}%")
    display_df['Demand %'] = display_df['Demand %'].apply(lambda x: f"{x:.1f}%")
    display_df['Supply %'] = display_df['Supply %'].apply(lambda x: f"{x:.1f}%")
    display_df['Gap (pp)'] = display_df['Gap (pp)'].apply(lambda x: f"{x:+.1f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def render_strategic_actions(engine):
    st.markdown("### 🎯 Strategic Actions")
    
    summary = engine.get_intent_summary()
    
    expand = summary[summary['gap'] > 3].sort_values('gap', ascending=False)
    rationalize = summary[summary['gap'] < -3].sort_values('gap', ascending=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style="background: #27AE60; color: white; padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                ⬆️ EXPAND (Under-served)
            </div>
        """, unsafe_allow_html=True)
        
        if len(expand) > 0:
            for _, row in expand.iterrows():
                st.markdown(f"""
                    <div style="background: white; border-radius: 12px; padding: 1rem; margin-bottom: 0.8rem; border-left: 4px solid #27AE60; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
                        <div style="display: flex; align-items: center; gap: 0.8rem;">
                            <span style="font-size: 2rem;">{row['icon']}</span>
                            <div style="flex: 1;">
                                <div style="font-weight: 600;">{row['intention_id']}: {row['name'][:40]}</div>
                                <div style="font-size: 0.8rem; color: #555;">
                                    Gap: <span style="color: #27AE60; font-weight: 600;">{row['gap']:+.1f}pp</span>
                                    | Demand: {row['demand_pct']:.1f}% | Supply: {row['supply_pct']:.1f}%
                                </div>
                                <div style="font-size: 0.8rem; color: #2E86C1;">
                                    Price: ${row['mean_price']:.4f} | Articles: {row['article_count']:,}
                                </div>
                            </div>
                        </div>
                        <div style="margin-top: 0.5rem; padding: 0.5rem; background: #f0f9f0; border-radius: 8px; font-size: 0.8rem; color: #1a6b3a;">
                            💡 <b>Recommendation:</b> Increase assortment by 10-15%.
                            {row['intention_id']} has {row['demand_pct']:.1f}% demand but only {row['supply_pct']:.1f}% supply.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No under-served intentions detected.")
    
    with col2:
        st.markdown("""
            <div style="background: #E74C3C; color: white; padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                ⬇️ RATIONALIZE (Over-supplied)
            </div>
        """, unsafe_allow_html=True)
        
        if len(rationalize) > 0:
            for _, row in rationalize.iterrows():
                st.markdown(f"""
                    <div style="background: white; border-radius: 12px; padding: 1rem; margin-bottom: 0.8rem; border-left: 4px solid #E74C3C; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
                        <div style="display: flex; align-items: center; gap: 0.8rem;">
                            <span style="font-size: 2rem;">{row['icon']}</span>
                            <div style="flex: 1;">
                                <div style="font-weight: 600;">{row['intention_id']}: {row['name'][:40]}</div>
                                <div style="font-size: 0.8rem; color: #555;">
                                    Gap: <span style="color: #E74C3C; font-weight: 600;">{row['gap']:+.1f}pp</span>
                                    | Demand: {row['demand_pct']:.1f}% | Supply: {row['supply_pct']:.1f}%
                                </div>
                                <div style="font-size: 0.8rem; color: #2E86C1;">
                                    Price: ${row['mean_price']:.4f} | Articles: {row['article_count']:,}
                                </div>
                            </div>
                        </div>
                        <div style="margin-top: 0.5rem; padding: 0.5rem; background: #fdf0f0; border-radius: 8px; font-size: 0.8rem; color: #8b1a1a;">
                            💡 <b>Recommendation:</b> Reduce assortment by 10-15%.
                            {row['intention_id']} has {row['supply_pct']:.1f}% supply but only {row['demand_pct']:.1f}% demand.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No over-supplied intentions detected.")

def render_recommendations(engine):
    st.markdown("### 🛍️ Product Recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        intent_filter = st.selectbox(
            "🎯 Filter by Intention",
            options=[-1] + list(range(10)),
            format_func=lambda x: "All Intentions" if x == -1 else f"{engine.get_intention_icon(x)} {engine.get_intention_name(x)[:30]}"
        )
    
    products = []
    for aid, intent in engine.article_intent_dict.items():
        if intent_filter == -1 or np.argmax(intent) == intent_filter:
            products.append((aid, intent[np.argmax(intent)]))
    
    products.sort(key=lambda x: x[1], reverse=True)
    products = products[:24]
    
    cols = st.columns(4)
    for idx, (article_id, score) in enumerate(products):
        # SỬA DÒNG NÀY: Lấy trực tiếp từ dictionary thay vì gọi method
        # details = engine.get_article_details(article_id)
        
        # Cách 1: Lấy từ article_meta_dict trực tiếp
        article_id_str = str(article_id)
        details = engine.article_meta_dict.get(article_id_str, {})
        
        if not details:  # Nếu không có details, bỏ qua
            continue
        
        with cols[idx % 4]:
            img_path = engine.get_image_path(article_id)
            intent = np.argmax(engine.article_intent_dict.get(str(article_id), np.zeros(10)))
            
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            
            if img_path and os.path.exists(img_path):
                try:
                    st.image(Image.open(img_path), use_container_width=True)
                except:
                    st.image("https://via.placeholder.com/300x400?text=H&M", use_container_width=True)
            else:
                st.image("https://via.placeholder.com/300x400?text=H&M", use_container_width=True)
            
            # Lấy tên sản phẩm từ details
            product_name = details.get('prod_name', 'Unknown')
            if pd.isna(product_name):
                product_name = 'Unknown'
            
            st.markdown(f"""
                <div class="product-info">
                    <div class="product-name">{product_name[:35]}</div>
                    <span class="intention-badge" style="background:{engine.get_intention_color(intent)};">
                        {engine.get_intention_icon(intent)} {engine.get_intention_name(intent)[:20]}
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# MAIN
# ============================================================================
def main():
    st.markdown("""
        <div class="main-header">
            <div>
                <h1>📊 H&M Strategic Business Intelligence</h1>
                <p>Real Data Analysis · Three-Tower Neural Network · AUC 0.8201</p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; opacity: 0.8;">Model AUC</div>
                <div style="font-size: 1.5rem; font-weight: 700;">0.8201</div>
                <div style="font-size: 0.7rem; opacity: 0.7;">+3.54% vs Two-Tower</div>
                <div style="font-size: 0.6rem; opacity: 0.6;">t = 103.12, p &lt; 0.001</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    data_dir = load_data()
    if data_dir is None:
        st.error("Failed to load data. Please check your connection.")
        return
    
    engine = BIEngine(data_dir)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🧠 Model Comparison",
        "📈 Supply-Demand",
        "📋 Strategic Actions",
        "🛍️ Recommendations"
    ])
    
    with tab1:
        # Dashboard tab - hiển thị tổng quan
        render_supply_demand_gap(engine, chart_key="tab1_gap")
        st.markdown("---")
        render_intention_stats(engine)
    
    with tab2:
        # Model Comparison tab
        render_model_comparison(engine)
    
    with tab3:
        # Supply-Demand tab - hiển thị chi tiết
        render_supply_demand_gap(engine, chart_key="tab3_gap")
        st.markdown("---")
        render_intention_stats(engine)
    
    with tab4:
        # Strategic Actions tab
        render_strategic_actions(engine)
    
    with tab5:
        # Recommendations tab
        render_recommendations(engine)
    
    st.markdown("""
        <div class="footer">
            <p>🏢 H&M Strategic BI · Real Data from 10% Test Set</p>
            <p>2,644 articles · 15,233 users · 27,915 interactions</p>
        </div>
    """, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()
