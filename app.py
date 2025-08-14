import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import nltk
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from scipy.sparse import hstack
import pickle
import os
import io
import docx
import PyPDF2
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# NLTK corpora/models required for improved perplexity
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')

from nltk.corpus import brown
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Ensure punkt is available
try:
    nltk.sent_tokenize("Test sentence.")
except LookupError:
    nltk.download('punkt')

# Page configuration
st.set_page_config(
    page_title="AI Content Detector Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/ai-content-detector-pro',
        'Report a bug': "https://github.com/yourusername/ai-content-detector-pro/issues",
        'About': "# AI Content Detector Pro\nAdvanced AI-generated content detection using multiple analysis techniques."
    }
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #ffffff;
        color: #333333;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #FF6B6B;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF5252;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Metric cards styling */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #333333;
    }
    
    /* Feature highlight styling */
    .feature-highlight {
        background-color: #e3f2fd;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        border-left: 4px solid #2196f3;
        color: #333333;
    }
    
    /* Sidebar styling - Target Streamlit sidebar specifically with dark theme */
    .css-1d391kg, .css-1lcbmhc, .css-17eq0hr {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border-right: 2px solid #334155 !important;
        padding: 20px !important;
        min-height: 100vh !important;
    }
    
    /* Ensure sidebar has proper background */
    .css-1d391kg > div, .css-1lcbmhc > div, .css-17eq0hr > div {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
    }
    
    /* Sidebar text elements - force white text */
    .css-1d391kg *, .css-1lcbmhc *, .css-17eq0hr * {
        color: #ffffff !important;
    }
    
    /* Sidebar title and headers - force light text */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3,
    .css-1lcbmhc h1, .css-1lcbmhc h2, .css-1lcbmhc h3,
    .css-17eq0hr h1, .css-17eq0hr h2, .css-17eq0hr h3 {
        color: #ffffff !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    }
    
    /* Sidebar markdown text - force white text */
    .css-1d391kg .stMarkdown, .css-1lcbmhc .stMarkdown, .css-17eq0hr .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Sidebar labels and text - force white text */
    .css-1d391kg label, .css-1lcbmhc label, .css-17eq0hr label,
    .css-1d391kg p, .css-1lcbmhc p, .css-17eq0hr p,
    .css-1d391kg span, .css-1lcbmhc span, .css-17eq0hr span {
        color: #ffffff !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #FF6B6B;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background-color: #ffffff;
        color: #333333;
        border: 2px solid #e9ecef;
        border-radius: 8px;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background-color: #ffffff;
        color: #333333;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #ffffff;
        color: #333333;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #333333;
    }
    
    /* Markdown text styling */
    .markdown-text-container {
        color: #333333;
        background-color: #ffffff;
    }
    
    /* Chart container styling */
    .js-plotly-plot {
        background-color: #ffffff;
    }
    
    /* Success/Error message styling */
    .stAlert {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #e9ecef;
    }
    
    /* Overall page background */
    .stApp {
        background-color: #ffffff;
        color: #333333;
    }
    
    /* Sidebar background */
    .css-1d391kg {
        background-color: #ffffff;
        color: #333333;
    }
    
    /* Ensure main content text is readable (scope to main only) */
    .main .block-container p,
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container h4,
    .main .block-container h5,
    .main .block-container h6,
    .main .block-container span,
    .main .block-container div {
        color: #1f2937 !important; /* slate-800 */
    }
    
    /* Streamlit default text color override (main only) */
    .main .block-container .stMarkdown {
        color: #1f2937 !important;
    }

    /* Stable sidebar targeting for consistent readability across versions */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important; /* slate-900 → slate-800 */
    }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stMultiSelect > div > div,
    section[data-testid="stSidebar"] .stTextInput > div > div,
    section[data-testid="stSidebar"] .stNumberInput > div > div {
        background-color: #334155 !important; /* slate-700 */
        color: #ffffff !important;
        border: 1px solid #475569 !important; /* slate-600 */
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > label,
    section[data-testid="stSidebar"] .stCheckbox > label,
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #e2e8f0 !important; /* slate-200 for better readability */
    }
    section[data-testid="stSidebar"] .stAlert {
        background-color: #334155 !important;
        color: #e2e8f0 !important;
        border: 1px solid #475569 !important;
    }
    
    /* Metric display styling - ensure strong contrast */
    .metric-value {
        color: #1f2937; /* slate-800 */
        font-weight: 700;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"] {
        color: #1f2937 !important;
        opacity: 1 !important;
    }
    
    /* Chart title styling */
    .gtitle {
        color: #333333 !important;
    }
    
    /* Axis labels styling */
    .g-gtitle, .g-xtitle, .g-ytitle {
        color: #333333 !important;
    }
    
    /* Streamlit container styling */
    .block-container {
        background-color: #ffffff;
        color: #333333;
    }
    
    /* Header styling */
    .main .block-container h1 {
        color: #333333 !important;
        border-bottom: 2px solid #FF6B6B;
        padding-bottom: 0.5rem;
    }
    
    /* Subheader styling */
    .main .block-container h2, .main .block-container h3 {
        color: #333333 !important;
    }
    
    /* Sidebar header styling - multiple CSS classes for compatibility */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3,
    .css-1lcbmhc h1, .css-1lcbmhc h2, .css-1lcbmhc h3,
    .css-17eq0hr h1, .css-17eq0hr h2, .css-17eq0hr h3 {
        color: #ffffff !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    }
    
    /* Sidebar text styling - multiple CSS classes for compatibility */
    .css-1d391kg p, .css-1d391kg span, .css-1d391kg div,
    .css-1lcbmhc p, .css-1lcbmhc span, .css-1lcbmhc div,
    .css-17eq0hr p, .css-17eq0hr span, .css-17eq0hr div {
        color: #f1f5f9 !important;
    }
    
    /* Sidebar selectbox styling - multiple CSS classes for compatibility */
    .css-1d391kg .stSelectbox > div > div,
    .css-1lcbmhc .stSelectbox > div > div,
    .css-17eq0hr .stSelectbox > div > div {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
        border: 2px solid #475569 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    /* Sidebar selectbox focus state */
    .css-1d391kg .stSelectbox > div > div:focus,
    .css-1lcbmhc .stSelectbox > div > div:focus,
    .css-17eq0hr .stSelectbox > div > div:focus {
        border-color: #FF6B6B !important;
        box-shadow: 0 0 0 0.2rem rgba(255, 107, 107, 0.4) !important;
        background-color: #475569 !important;
    }
    
    /* Sidebar selectbox label styling */
    .css-1d391kg .stSelectbox > label,
    .css-1lcbmhc .stSelectbox > label,
    .css-17eq0hr .stSelectbox > label {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    }
    
    /* Sidebar checkbox container styling */
    .css-1d391kg .stCheckbox,
    .css-1lcbmhc .stCheckbox,
    .css-17eq0hr .stCheckbox {
        margin-bottom: 15px !important;
        padding: 8px 0 !important;
    }
    
    /* Sidebar checkbox styling - multiple CSS classes for compatibility */
    .css-1d391kg .stCheckbox > label,
    .css-1lcbmhc .stCheckbox > label,
    .css-17eq0hr .stCheckbox > label {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        line-height: 1.4 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    }
    
    /* Sidebar checkbox input styling */
    .css-1d391kg .stCheckbox input[type="checkbox"],
    .css-1lcbmhc .stCheckbox input[type="checkbox"],
    .css-17eq0hr .stCheckbox input[type="checkbox"] {
        accent-color: #FF6B6B !important;
        transform: scale(1.2) !important;
        filter: brightness(1.2) !important;
    }
    
    /* Sidebar markdown styling - multiple CSS classes for compatibility */
    .css-1d391kg .stMarkdown,
    .css-1lcbmhc .stMarkdown,
    .css-17eq0hr .stMarkdown {
        color: #f1f5f9 !important;
    }
    
    /* Sidebar success/error messages - multiple CSS classes for compatibility */
    .css-1d391kg .stSuccess, .css-1d391kg .stError, .css-1d391kg .stWarning, .css-1d391kg .stInfo,
    .css-1lcbmhc .stSuccess, .css-1lcbmhc .stError, .css-1lcbmhc .stWarning, .css-1lcbmhc .stInfo,
    .css-17eq0hr .stSuccess, .css-17eq0hr .stError, .css-17eq0hr .stWarning, .css-17eq0hr .stInfo {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        margin: 8px 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    /* Specific message type styling in sidebar */
    .css-1d391kg .stSuccess, .css-1lcbmhc .stSuccess, .css-17eq0hr .stSuccess {
        background-color: #065f46 !important;
        border-color: #047857 !important;
        color: #ecfdf5 !important;
    }
    
    .css-1d391kg .stInfo, .css-1lcbmhc .stInfo, .css-17eq0hr .stInfo {
        background-color: #0c4a6e !important;
        border-color: #0369a1 !important;
        color: #f0f9ff !important;
    }
    
    /* Sidebar divider styling - multiple CSS classes for compatibility */
    .css-1d391kg hr, .css-1lcbmhc hr, .css-17eq0hr hr {
        border-color: #475569 !important;
        margin: 1rem 0 !important;
        border-width: 2px !important;
    }
    
    /* Additional sidebar styling for better readability */
    .css-1d391kg, .css-1lcbmhc, .css-17eq0hr {
        padding: 20px !important;
        position: relative !important;
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%) !important;
    }
    
    /* Sidebar content spacing */
    .css-1d391kg > *, .css-1lcbmhc > *, .css-17eq0hr > * {
        margin-bottom: 15px !important;
    }
    
    /* Sidebar text elements spacing */
    .css-1d391kg p, .css-1lcbmhc p, .css-17eq0hr p {
        margin-bottom: 10px !important;
        line-height: 1.6 !important;
    }
    
    /* Force sidebar text color override for dark theme */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3,
    .css-1lcbmhc h1, .css-1lcbmhc h2, .css-1lcbmhc h3,
    .css-17eq0hr h1, .css-17eq0hr h2, .css-17eq0hr h3,
    .css-1d391kg p, .css-1lcbmhc p, .css-17eq0hr p,
    .css-1d391kg span, .css-1lcbmhc span, .css-17eq0hr span,
    .css-1d391kg label, .css-1lcbmhc label, .css-17eq0hr label {
        color: #ffffff !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    }
    
    /* Additional aggressive text color override */
    .css-1d391kg div, .css-1lcbmhc div, .css-17eq0hr div {
        color: #ffffff !important;
    }
    
    /* Override any remaining dark text */
    .css-1d391kg *, .css-1lcbmhc *, .css-17eq0hr * {
        color: #ffffff !important;
    }
    
    /* Force all sidebar text to be white - final override */
    .css-1d391kg, .css-1lcbmhc, .css-17eq0hr,
    .css-1d391kg *, .css-1lcbmhc *, .css-17eq0hr * {
        color: #ffffff !important;
    }
    
    /* Specific override for any stubborn elements */
    .css-1d391kg div, .css-1lcbmhc div, .css-17eq0hr div,
    .css-1d391kg span, .css-1lcbmhc span, .css-17eq0hr span,
    .css-1d391kg p, .css-1lcbmhc p, .css-17eq0hr p {
        color: #ffffff !important;
    }
    
    /* Force sidebar markdown text to be white */
    .css-1d391kg .stMarkdown p, .css-1lcbmhc .stMarkdown p, .css-17eq0hr .stMarkdown p,
    .css-1d391kg .stMarkdown strong, .css-1lcbmhc .stMarkdown strong, .css-17eq0hr .stMarkdown strong,
    .css-1d391kg .stMarkdown b, .css-1lcbmhc .stMarkdown b, .css-17eq0hr .stMarkdown b {
        color: #ffffff !important;
    }
    
    /* Force sidebar help text to be white */
    .css-1d391kg .stSelectbox > div > div > div,
    .css-1lcbmhc .stSelectbox > div > div > div,
    .css-17eq0hr .stSelectbox > div > div > div {
        color: #ffffff !important;
    }
    
    /* Force sidebar title to be white */
    .css-1d391kg h1, .css-1lcbmhc h1, .css-17eq0hr h1 {
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: bold !important;
        margin-bottom: 20px !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5) !important;
    }
    
    /* Force sidebar headers to be white */
    .css-1d391kg h3, .css-1lcbmhc h3, .css-17eq0hr h3 {
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.4) !important;
    }
    
    /* Force sidebar markdown content to be white */
    .css-1d391kg .stMarkdown, .css-1lcbmhc .stMarkdown, .css-17eq0hr .stMarkdown {
        color: #ffffff !important;
        line-height: 1.6 !important;
        background-color: transparent !important;
    }
    
    /* Force sidebar markdown paragraphs to be white */
    .css-1d391kg .stMarkdown p, .css-1lcbmhc .stMarkdown p, .css-17eq0hr .stMarkdown p {
        color: #ffffff !important;
        margin-bottom: 8px !important;
        line-height: 1.5 !important;
    }
    
    /* Force sidebar markdown lists to be white */
    .css-1d391kg .stMarkdown ul, .css-1lcbmhc .stMarkdown ul, .css-17eq0hr .stMarkdown ul,
    .css-1d391kg .stMarkdown ol, .css-1lcbmhc .stMarkdown ol, .css-17eq0hr .stMarkdown ol {
        color: #ffffff !important;
        margin-left: 20px !important;
    }
    
    /* Force sidebar markdown list items to be white */
    .css-1d391kg .stMarkdown li, .css-1lcbmhc .stMarkdown li, .css-17eq0hr .stMarkdown li {
        color: #ffffff !important;
        margin-bottom: 5px !important;
    }
    
    /* Force sidebar markdown strong/bold text to be white */
    .css-1d391kg .stMarkdown strong, .css-1lcbmhc .stMarkdown strong, .css-17eq0hr .stMarkdown strong {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Force sidebar divider to be visible */
    .css-1d391kg hr, .css-1lcbmhc hr, .css-17eq0hr hr {
        border-color: #64748b !important;
        border-width: 2px !important;
        margin: 20px 0 !important;
        opacity: 0.8 !important;
    }
    
    /* Input field focus styling */
    .stTextArea textarea:focus {
        border-color: #FF6B6B !important;
        box-shadow: 0 0 0 0.2rem rgba(255, 107, 107, 0.25) !important;
    }
    
    /* Selectbox focus styling */
    .stSelectbox > div > div:focus {
        border-color: #FF6B6B !important;
        box-shadow: 0 0 0 0.2rem rgba(255, 107, 107, 0.25) !important;
    }
    
    /* Alerts - main content (high contrast, rounded) */
    .main .block-container .stAlert {
        border-radius: 12px !important;
        padding: 14px 16px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        border-width: 1px !important;
    }
    .main .block-container .stSuccess {
        background-color: #ecfdf5 !important; /* emerald-50 */
        color: #065f46 !important;           /* emerald-800 */
        border: 1px solid #34d399 !important; /* emerald-400 */
        font-weight: 600 !important;
    }
    .main .block-container .stError {
        background-color: #fef2f2 !important; /* red-50 */
        color: #991b1b !important;            /* red-800 */
        border: 1px solid #fca5a5 !important; /* red-300 */
        font-weight: 600 !important;
    }
    .main .block-container .stWarning {
        background-color: #fffbeb !important; /* amber-50 */
        color: #92400e !important;            /* amber-800 */
        border: 1px solid #fbbf24 !important; /* amber-400 */
        font-weight: 600 !important;
    }
    .main .block-container .stInfo {
        background-color: #eff6ff !important; /* blue-50 */
        color: #1e3a8a !important;            /* blue-800 */
        border: 1px solid #93c5fd !important; /* blue-300 */
        font-weight: 600 !important;
    }

    /* Alerts - sidebar (rounded, subtle, readable on dark) */
    section[data-testid="stSidebar"] .stAlert {
        border-radius: 10px !important;
        padding: 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
        border-width: 1px !important;
    }
    section[data-testid="stSidebar"] .stSuccess {
        background-color: rgba(16, 185, 129, 0.15) !important; /* emerald-500 @ 15% */
        color: #e2e8f0 !important; /* slate-200 */
        border: 1px solid rgba(16, 185, 129, 0.45) !important;
    }
    section[data-testid="stSidebar"] .stError {
        background-color: rgba(239, 68, 68, 0.15) !important; /* red-500 @ 15% */
        color: #e2e8f0 !important;
        border: 1px solid rgba(239, 68, 68, 0.45) !important;
    }
    section[data-testid="stSidebar"] .stWarning {
        background-color: rgba(245, 158, 11, 0.18) !important; /* amber-500 */
        color: #e2e8f0 !important;
        border: 1px solid rgba(245, 158, 11, 0.5) !important;
    }
    section[data-testid="stSidebar"] .stInfo {
        background-color: rgba(59, 130, 246, 0.18) !important; /* blue-500 */
        color: #e2e8f0 !important;
        border: 1px solid rgba(59, 130, 246, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

class AIContentDetector:
    def __init__(self):
        # Stronger default text features and a linear classifier that works well on sparse TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=True
        )
        self.classifier = LogisticRegression(
            solver='liblinear', random_state=42, max_iter=1000, class_weight='balanced'
        )
        self.model_path = "ai_detector_model.pkl"
        self.threshold = 0.5
        self.metrics = {}
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load existing model or train a new one."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.vectorizer = model_data['vectorizer']
                    self.classifier = model_data['classifier']
                    # Optional fields for newer models
                    self.threshold = model_data.get('threshold', 0.5)
                    self.metrics = model_data.get('metrics', {})
                st.sidebar.success("Model loaded successfully")
            except:
                st.sidebar.warning("Model loading failed, training new model...")
                self.train_model()
        else:
            st.sidebar.info("Training new model...")
            self.train_model()
    
    def train_model(self):
        """Train the ML model with synthetic data.
        Note: For best real-world performance, replace this synthetic dataset
        with a labeled dataset of real human and AI texts.
        """
        # Generate synthetic training data (larger, with more variety)
        human_texts = self.generate_human_like_texts(300)
        ai_texts = self.generate_ai_like_texts(300)

        # Combine and label data
        texts = human_texts + ai_texts
        labels = [0] * len(human_texts) + [1] * len(ai_texts)  # 0=human, 1=AI

        # Train/validation split
        X_train_texts, X_val_texts, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Vectorize
        X_train = self.vectorizer.fit_transform(X_train_texts)
        X_val = self.vectorizer.transform(X_val_texts)

        # Train classifier
        self.classifier.fit(X_train, y_train)

        # Validation metrics and threshold tuning (Youden's J)
        try:
            val_proba = self.classifier.predict_proba(X_val)[:, 1]
        except Exception:
            # Fallback for classifiers without predict_proba
            val_proba = self.classifier.decision_function(X_val)
            # Min-max normalize to [0,1]
            val_proba = (val_proba - val_proba.min()) / (val_proba.max() - val_proba.min() + 1e-9)

        fpr, tpr, thresholds = roc_curve(y_val, val_proba)
        youden_j = tpr - fpr
        best_idx = int(youden_j.argmax())
        self.threshold = float(thresholds[best_idx])

        val_pred = (val_proba >= self.threshold).astype(int)
        acc = float(accuracy_score(y_val, val_pred))
        try:
            auc = float(roc_auc_score(y_val, val_proba))
        except Exception:
            auc = None

        self.metrics = {
            'val_accuracy': acc,
            'val_auc': auc,
            'threshold': self.threshold,
            'num_train_samples': len(X_train_texts),
            'num_val_samples': len(X_val_texts)
        }

        # Save model
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'threshold': self.threshold,
            'metrics': self.metrics,
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

        st.sidebar.success(f"Model trained and saved | Val Acc: {acc:.2f}" + (f", AUC: {auc:.2f}" if auc is not None else ""))
    
    def generate_human_like_texts(self, num_samples: int) -> List[str]:
        """Generate human-like text samples for training."""
        human_patterns = [
            "I think that this is really interesting because it shows how people naturally write with varying sentence lengths and sometimes make small mistakes or use informal language.",
            "The weather today was quite unpredictable. It started sunny but then clouds rolled in and it began to rain. I forgot my umbrella at home, which was pretty annoying.",
            "My friend told me about this new restaurant downtown. The food was amazing, though a bit expensive. We had to wait like 20 minutes for a table, but it was totally worth it.",
            "I'm not sure if I should take that job offer. The salary is good, but the commute would be really long. Plus, I'd have to move to a different city.",
            "Yesterday I went to the store to buy some groceries. I got bread, milk, eggs, and some vegetables. The cashier was really friendly and we chatted for a bit.",
            "This book is really hard to put down. The characters are so well-developed and the plot has so many unexpected twists. I stayed up until 2 AM reading it.",
            "I tried cooking that new recipe I found online. It didn't turn out exactly like the picture, but it still tasted pretty good. I think I used too much salt though.",
            "The movie was okay, I guess. The special effects were cool, but the story was kind of predictable. My friend loved it though, so maybe it's just me.",
            "I need to remember to call the dentist tomorrow to schedule my appointment. I've been putting it off for weeks now, which is probably not a good idea.",
            "That meeting was so boring. The presenter just kept talking and talking without really saying anything important. I almost fell asleep."
        ]
        
        # Generate variations
        texts = []
        for _ in range(num_samples):
            base_text = np.random.choice(human_patterns)
            # Add some variation
            if np.random.random() > 0.5:
                base_text += " " + np.random.choice([
                    "Anyway, that's what happened.",
                    "I'm not really sure about that.",
                    "What do you think?",
                    "It was kind of weird.",
                    "I guess we'll see."
                ])
            texts.append(base_text)
        
        return texts
    
    def generate_ai_like_texts(self, num_samples: int) -> List[str]:
        """Generate AI-like text samples for training."""
        ai_patterns = [
            "The implementation of this system demonstrates a comprehensive approach to data analysis, incorporating multiple methodologies to ensure accurate results.",
            "Furthermore, the analysis reveals significant correlations between the variables under investigation, suggesting a robust relationship that warrants further examination.",
            "In conclusion, the findings presented herein provide compelling evidence for the hypothesis, supported by rigorous statistical analysis and peer-reviewed research.",
            "The methodology employed in this study follows established protocols and best practices, ensuring reproducibility and reliability of the experimental outcomes.",
            "Additionally, the results indicate a clear pattern of behavior that aligns with theoretical predictions, thereby validating the underlying assumptions of the model.",
            "The comprehensive review of the literature reveals a consensus among researchers regarding the fundamental principles governing this phenomenon.",
            "Moreover, the data analysis demonstrates statistical significance across multiple dimensions, providing strong support for the proposed framework.",
            "The systematic approach adopted in this investigation ensures methodological rigor while maintaining practical applicability in real-world scenarios.",
            "Furthermore, the empirical evidence presented herein contributes to the growing body of knowledge in this field, offering valuable insights for future research.",
            "The theoretical framework provides a solid foundation for understanding the complex interactions between various factors influencing the observed outcomes."
        ]
        
        # Generate variations
        texts = []
        for _ in range(num_samples):
            base_text = np.random.choice(ai_patterns)
            # Add some variation
            if np.random.random() > 0.5:
                base_text += " " + np.random.choice([
                    "This conclusion is supported by extensive empirical evidence.",
                    "Further research is warranted to explore these findings.",
                    "The implications of these results are far-reaching.",
                    "This analysis provides valuable insights for practitioners.",
                    "The methodology ensures robust and reliable outcomes."
                ])
            texts.append(base_text)
        
        return texts

class TextAnalyzer:
    def __init__(self):
        self.detector = AIContentDetector()
        # Expose validation metrics in the UI if available
        if self.detector.metrics:
            with st.sidebar.expander("Model Metrics", expanded=False):
                st.write({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in self.detector.metrics.items()})
    
    def extract_text_from_file(self, uploaded_file) -> str:
        """Extract text from uploaded file."""
        try:
            if uploaded_file.type == "text/plain":
                return uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(uploaded_file)
                return " ".join([paragraph.text for paragraph in doc.paragraphs])
            elif uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            else:
                return uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return ""
    
    def calculate_stylometric_features(self, text: str) -> Dict[str, float]:
        """Calculate comprehensive stylometric features."""
        # Clean and tokenize text
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text.lower())
        
        # Remove punctuation and numbers
        words = [word for word in words if word.isalpha()]
        
        if not words or not sentences:
            return {}
        
        # Basic features
        avg_sentence_length = np.mean([len(nltk.word_tokenize(s)) for s in sentences])
        vocab_richness = len(set(words)) / len(words)
        
        # Word frequency analysis
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(10)
        common_words_ratio = sum(freq for word, freq in most_common_words) / len(words)
        
        # Sentence complexity
        sentence_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
        sentence_length_variance = np.var(sentence_lengths)
        
        # Word length analysis
        word_lengths = [len(word) for word in words]
        avg_word_length = np.mean(word_lengths)
        word_length_variance = np.var(word_lengths)
        
        # Punctuation analysis
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        punctuation_ratio = punctuation_count / len(words)
        
        # Capitalization analysis
        capital_letters = len(re.findall(r'[A-Z]', text))
        capitalization_ratio = capital_letters / len(text.replace(" ", ""))
        
        # Unique word ratio
        unique_word_ratio = len(set(words)) / len(words)
        
        return {
            "avg_sentence_length": avg_sentence_length,
            "vocab_richness": vocab_richness,
            "common_words_ratio": common_words_ratio,
            "sentence_length_variance": sentence_length_variance,
            "avg_word_length": avg_word_length,
            "word_length_variance": word_length_variance,
            "punctuation_ratio": punctuation_ratio,
            "capitalization_ratio": capitalization_ratio,
            "unique_word_ratio": unique_word_ratio
        }
    
    def calculate_perplexity_score(self, text: str) -> Dict[str, float]:
        """Improved perplexity analysis.

        - Uses a cached Kneser–Ney trigram LM on Brown corpus.
        - Computes raw perplexity (ppl) and a burstiness measure
          (variance of per-sentence average log-probabilities).
        - Returns a dict with: { 'score', 'ppl', 'burstiness' }.

        Heuristic mapping to human-likeness:
          - Higher ppl often corresponds to less template-like, more human text.
          - Higher burstiness (variability across sentences) also suggests human text.
        """
        try:
            sentences = nltk.sent_tokenize(text)
            tokens = [w.lower() for w in nltk.word_tokenize(text) if any(c.isalpha() for c in w)]
            if len(tokens) < 20 or len(sentences) < 2:
                return { 'score': 0.5, 'ppl': 0.0, 'burstiness': 0.0 }

            # Train or reuse LM
            if not hasattr(self, '_lm'):
                train_sents = [[w.lower() for w in sent] for sent in brown.sents()]
                train_data, vocab = padded_everygram_pipeline(3, train_sents)
                self._lm = KneserNeyInterpolated(3)
                self._lm.fit(train_data, vocab)

            # Token-level perplexity
            padded = list(pad_both_ends(tokens, n=3))
            test_grams = [tuple(padded[i-2:i+1]) for i in range(2, len(padded))]
            log_prob = 0.0
            log_probs = []
            for gram in test_grams:
                context = gram[:-1]
                word = gram[-1]
                prob = self._lm.score(word, context)
                lp = math.log(max(prob, 1e-12))
                log_prob += lp
                log_probs.append(lp)
            ppl = math.exp(-log_prob / max(len(test_grams), 1))

            # Sentence-level burstiness: variance of mean log-probs per sentence
            sent_scores = []
            idx = 0
            tokenized_sents = [
                [w.lower() for w in nltk.word_tokenize(s) if any(c.isalpha() for c in w)]
                for s in sentences
            ]
            for sent in tokenized_sents:
                if len(sent) < 2:
                    continue
                sp = list(pad_both_ends(sent, n=3))
                grams = [tuple(sp[i-2:i+1]) for i in range(2, len(sp))]
                if not grams:
                    continue
                s_log = 0.0
                for g in grams:
                    s_log += math.log(max(self._lm.score(g[-1], g[:-1]), 1e-12))
                sent_scores.append(s_log / len(grams))
            burstiness = float(np.var(sent_scores)) if len(sent_scores) >= 2 else 0.0

            # Normalize to [0,1]
            ppl_component = min(ppl / 600.0, 1.0)  # cap around 600
            burst_component = max(0.0, min((burstiness + 5.0) / 10.0, 1.0))  # widen typical range
            score = 0.6 * ppl_component + 0.4 * burst_component

            return {
                'score': float(max(0.0, min(1.0, score))),
                'ppl': float(ppl),
                'burstiness': float(burstiness),
            }
        except Exception:
            return { 'score': 0.5, 'ppl': 0.0, 'burstiness': 0.0 }
    
    def get_ml_prediction(self, text: str) -> float:
        """Get ML model prediction."""
        try:
            # Vectorize text
            X = self.detector.vectorizer.transform([text])
            # Get prediction probability
            if hasattr(self.detector.classifier, 'predict_proba'):
                prob = self.detector.classifier.predict_proba(X)[0, 1]
            else:
                score = self.detector.classifier.decision_function(X)[0]
                # Min-max normalize single score to [0,1] with a logistic proxy
                prob = 1 / (1 + math.exp(-score))
            return float(prob)  # Probability of being AI-generated
        except Exception as e:
            st.warning(f"ML prediction failed: {str(e)}")
            return 0.5
    
    def analyze_text(self, text: str, method: str) -> Dict[str, Any]:
        """Comprehensive text analysis."""
        results = {}
        
        if method in ["Stylometric Analysis", "Combined Analysis"]:
            features = self.calculate_stylometric_features(text)
            results["stylometric"] = features
            
            # Calculate stylometric score
            stylometric_score = (
                (1 - features.get("common_words_ratio", 0)) * 0.3 +
                features.get("unique_word_ratio", 0) * 0.3 +
                (1 - features.get("punctuation_ratio", 0)) * 0.2 +
                features.get("sentence_length_variance", 0) / 100 * 0.2
            )
            results["stylometric_score"] = stylometric_score
        
        if method in ["Perplexity Analysis", "Combined Analysis"]:
            perplexity_details = self.calculate_perplexity_score(text)
            results["perplexity"] = perplexity_details.get('score', 0.5)
            results["perplexity_details"] = perplexity_details
        
        if method in ["ML Classification", "Combined Analysis"]:
            ml_score = self.get_ml_prediction(text)
            results["ml_score"] = ml_score
            # Also expose thresholded class for transparency
            if hasattr(self.detector, 'threshold'):
                results["ml_pred"] = int(ml_score >= self.detector.threshold)
        
        # Combined score calculation
        if method == "Combined Analysis":
            weights = {
                "stylometric": 0.25,
                "perplexity": 0.35,
                "ml": 0.40
            }
            
            combined_score = (
                results.get("stylometric_score", 0.5) * weights["stylometric"] +
                results.get("perplexity", 0.5) * weights["perplexity"] +
                (1 - results.get("ml_score", 0.5)) * weights["ml"]  # Invert ML score
            )
            results["combined_score"] = combined_score
        
        return results

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return TextAnalyzer()

analyzer = get_analyzer()

# Sidebar
with st.sidebar:
    st.title("Detection Settings")
    detection_method = st.selectbox(
        "Detection Method",
        ["Combined Analysis", "Stylometric Analysis", "Perplexity Analysis", "ML Classification"],
        help="Choose the analysis method. Combined Analysis uses all methods for best accuracy."
    )
    
    show_explanation = st.checkbox("Show Detailed Explanation", value=True)
    show_features = st.checkbox("Show Feature Breakdown", value=True)
    
    st.markdown("---")
    st.markdown("### Analysis Methods")
    st.markdown("""
    **Combined Analysis**: Uses all methods for highest accuracy
    **Stylometric**: Analyzes writing style patterns
    **Perplexity**: Measures text complexity
    **ML Classification**: Machine learning-based detection
    """)
    
    st.markdown("---")
    st.markdown("### Model Status")
    if os.path.exists("ai_detector_model.pkl"):
        st.markdown("<div class='stSuccess'>✅ <strong>ML Model Ready</strong></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='stInfo'>⏳ <strong>Training ML Model...</strong></div>", unsafe_allow_html=True)

# Main content
st.title("AI Content Detector Pro")
st.markdown("Advanced AI-generated content detection using multiple analysis techniques")

# File upload section
st.markdown("### Upload Document")
uploaded_file = st.file_uploader(
    "Upload a document (.txt, .docx, .pdf)",
    type=["txt", "docx", "pdf"],
    help="Supported formats: Text files, Word documents, PDF files"
)

# Text input section
st.markdown("### Or Enter Text Directly")
text_input = st.text_area(
    "Paste your text here",
    height=200,
    placeholder="Enter the text you want to analyze for AI-generated content detection..."
)

# Analysis button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("Analyze Content", use_container_width=True)

if analyze_button:
    if text_input or uploaded_file:
        # Get text content
        if uploaded_file:
            text_content = analyzer.extract_text_from_file(uploaded_file)
            if not text_content.strip():
                st.error("Could not extract text from the uploaded file.")
                st.stop()
        else:
            text_content = text_input
        
        if not text_content.strip():
            st.error("Please provide text to analyze.")
            st.stop()
        
        # Perform analysis
        with st.spinner("Analyzing content with advanced algorithms..."):
            results = analyzer.analyze_text(text_content, detection_method)
            
            # Display results
            st.markdown("## Analysis Results")
            
            # Create main results display
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Determine score based on method
                if detection_method == "Combined Analysis":
                    score = results.get("combined_score", 0.5)
                elif detection_method == "ML Classification":
                    score = 1 - results.get("ml_score", 0.5)
                elif detection_method == "Perplexity Analysis":
                    score = results.get("perplexity", 0.5)
                else:
                    score = results.get("stylometric_score", 0.5)
                
                human_probability = score * 100
                ai_probability = 100 - human_probability
                
                # Create enhanced pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=['Human-Written', 'AI-Generated'],
                    values=[human_probability, ai_probability],
                    hole=0.4,
                    marker_colors=['#00CC96', '#EF553B'],
                    textinfo='label+percent',
                    textfont_size=14
                )])
                
                fig.update_layout(
                    title="Content Origin Probability",
                    showlegend=True,
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#333333'),
                    title_font=dict(color='#333333')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Confidence Scores")
                
                # Create metric cards
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Human-Written</h4>
                    <h2 style="color: #00CC96;">{human_probability:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>AI-Generated</h4>
                    <h2 style="color: #EF553B;">{ai_probability:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence level
                if human_probability > 70:
                    confidence = "High"
                    color = "#00CC96"
                elif human_probability > 50:
                    confidence = "Medium"
                    color = "#FFA500"
                else:
                    confidence = "Low"
                    color = "#EF553B"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Confidence Level</h4>
                    <h2 style="color: {color};">{confidence}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed analysis section
            if show_explanation:
                st.markdown("## Detailed Analysis")
                
                # Create subplots for detailed analysis
                if detection_method in ["Stylometric Analysis", "Combined Analysis"] and "stylometric" in results:
                    st.markdown("### Stylometric Analysis")
                    
                    features = results["stylometric"]
                    
                    # Create feature comparison chart
                    feature_names = list(features.keys())
                    feature_values = list(features.values())
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=feature_names,
                            y=feature_values,
                            marker_color='#636EFA',
                            text=[f'{v:.3f}' for v in feature_values],
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Stylometric Features",
                        xaxis_title="Features",
                        yaxis_title="Values",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#333333'),
                        title_font=dict(color='#333333'),
                        xaxis=dict(gridcolor='#e9ecef'),
                        yaxis=dict(gridcolor='#e9ecef')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if show_features:
                        st.markdown("#### Feature Breakdown")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="feature-highlight">
                                <strong>Avg Sentence Length:</strong> {features.get('avg_sentence_length', 0):.2f} words
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="feature-highlight">
                                <strong>Vocabulary Richness:</strong> {features.get('vocab_richness', 0):.3f}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="feature-highlight">
                                <strong>Word Length Variance:</strong> {features.get('word_length_variance', 0):.2f}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="feature-highlight">
                                <strong>Punctuation Ratio:</strong> {features.get('punctuation_ratio', 0):.3f}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="feature-highlight">
                                <strong>Capitalization Ratio:</strong> {features.get('capitalization_ratio', 0):.3f}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="feature-highlight">
                                <strong>Unique Word Ratio:</strong> {features.get('unique_word_ratio', 0):.3f}
                            </div>
                            """, unsafe_allow_html=True)
                
                if detection_method in ["Perplexity Analysis", "Combined Analysis"] and "perplexity" in results:
                    st.markdown("### Perplexity Analysis")
                    
                    perplexity_score = results["perplexity"]
                    details = results.get("perplexity_details", {})
                    
                    # Create perplexity gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=perplexity_score * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Perplexity Score"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#636EFA"},
                            'steps': [
                                {'range': [0, 30], 'color': "#EF553B"},
                                {'range': [30, 70], 'color': "#FFA500"},
                                {'range': [70, 100], 'color': "#00CC96"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=300,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#333333'),
                        title_font=dict(color='#333333')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Show raw metrics for transparency
                    st.markdown(
                        f"<div class='metric-card'><p><strong>Raw Perplexity:</strong> {details.get('ppl', 0):.1f}</p>"
                        f"<p><strong>Burstiness:</strong> {details.get('burstiness', 0):.3f}</p></div>",
                        unsafe_allow_html=True
                    )
                
                if detection_method in ["ML Classification", "Combined Analysis"] and "ml_score" in results:
                    st.markdown("### Machine Learning Analysis")
                    
                    ml_score = results["ml_score"]
                    human_ml_prob = (1 - ml_score) * 100
                    ai_ml_prob = ml_score * 100
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>ML Model Prediction</h4>
                        <p><strong>Human Probability:</strong> {human_ml_prob:.1f}%</p>
                        <p><strong>AI Probability:</strong> {ai_ml_prob:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Summary and recommendations
            st.markdown("## Analysis Summary")
            
            if human_probability > 70:
                st.markdown("<div class='stSuccess'>High confidence that this content was written by a human.</div>", unsafe_allow_html=True)
                st.markdown("The text shows natural variations in writing style, vocabulary usage, and sentence structure typical of human writing.")
            elif human_probability > 50:
                st.markdown("<div class='stWarning'>Moderate confidence - mixed signals detected.</div>", unsafe_allow_html=True)
                st.markdown("The analysis shows some characteristics of both human and AI writing. Consider additional context or manual review.")
            else:
                st.markdown("<div class='stError'>High confidence that this content was AI-generated.</div>", unsafe_allow_html=True)
                st.markdown("The text exhibits patterns commonly associated with AI-generated content, such as consistent structure and vocabulary usage.")
            
            # Text statistics (force visible by avoiding dimmed metric style)
            st.markdown("### Text Statistics")
            word_count = len(text_content.split())
            char_count = len(text_content)
            sentence_count = len(nltk.sent_tokenize(text_content))

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-card'><h4>Words</h4><h2 class='metric-value'>{word_count:,}</h2></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-card'><h4>Characters</h4><h2 class='metric-value'>{char_count:,}</h2></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='metric-card'><h4>Sentences</h4><h2 class='metric-value'>{sentence_count:,}</h2></div>", unsafe_allow_html=True)
    
    else:
        st.error("Please provide text or upload a file to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>AI Content Detector Pro | Built with Streamlit</p>
    <p>Advanced AI-generated content detection using multiple analysis techniques</p>
</div>
""", unsafe_allow_html=True)