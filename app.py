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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import io
import docx
import PyPDF2
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

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
    page_icon="🔍",
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
    
    /* Sidebar styling */
    .css-1d391kg {
        padding: 2rem;
        background-color: #ffffff;
        color: #333333;
        border-right: 1px solid #e9ecef;
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
    
    /* Ensure all text is readable */
    p, h1, h2, h3, h4, h5, h6, span, div {
        color: #333333 !important;
    }
    
    /* Streamlit default text color override */
    .stMarkdown {
        color: #333333 !important;
    }
    
    /* Metric display styling */
    .metric-value {
        color: #333333;
        font-weight: bold;
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
    
    /* Sidebar header styling */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #333333 !important;
    }
    
    /* Sidebar text styling */
    .css-1d391kg p, .css-1d391kg span, .css-1d391kg div {
        color: #333333 !important;
    }
    
    /* Sidebar selectbox styling */
    .css-1d391kg .stSelectbox > div > div {
        background-color: #ffffff;
        color: #333333;
        border: 2px solid #e9ecef;
        border-radius: 8px;
    }
    
    /* Sidebar checkbox styling */
    .css-1d391kg .stCheckbox > label {
        color: #333333 !important;
        font-weight: 500;
    }
    
    /* Sidebar markdown styling */
    .css-1d391kg .stMarkdown {
        color: #333333 !important;
    }
    
    /* Sidebar success/error messages */
    .css-1d391kg .stSuccess,
    .css-1d391kg .stError,
    .css-1d391kg .stWarning,
    .css-1d391kg .stInfo {
        background-color: #ffffff !important;
        color: #333333 !important;
        border: 1px solid #e9ecef !important;
    }
    
    /* Sidebar divider styling */
    .css-1d391kg hr {
        border-color: #e9ecef;
        margin: 1rem 0;
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
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
        border: 1px solid #c3e6cb !important;
    }
    
    /* Error message styling */
    .stError {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        border: 1px solid #f5c6cb !important;
    }
    
    /* Warning message styling */
    .stWarning {
        background-color: #fff3cd !important;
        color: #856404 !important;
        border: 1px solid #ffeaa7 !important;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: #d1ecf1 !important;
        color: #0c5460 !important;
        border: 1px solid #bee5eb !important;
    }
</style>
""", unsafe_allow_html=True)

class AIContentDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_path = "ai_detector_model.pkl"
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load existing model or train a new one."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.vectorizer = model_data['vectorizer']
                    self.classifier = model_data['classifier']
                st.sidebar.success("✅ Model loaded successfully")
            except:
                st.sidebar.warning("⚠️ Model loading failed, training new model...")
                self.train_model()
        else:
            st.sidebar.info("🔄 Training new model...")
            self.train_model()
    
    def train_model(self):
        """Train the ML model with synthetic data."""
        # Generate synthetic training data
        human_texts = self.generate_human_like_texts(100)
        ai_texts = self.generate_ai_like_texts(100)
        
        # Combine and label data
        texts = human_texts + ai_texts
        labels = [0] * len(human_texts) + [1] * len(ai_texts)  # 0=human, 1=AI
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Train classifier
        self.classifier.fit(X, labels)
        
        # Save model
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        st.sidebar.success("✅ Model trained and saved")
    
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
    
    def calculate_perplexity_score(self, text: str) -> float:
        """Calculate perplexity-based score."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Calculate word frequency
        word_freq = Counter(words)
        total_words = len(words)
        
        # Calculate perplexity (simplified version)
        log_prob = 0
        for word in words:
            prob = word_freq[word] / total_words
            log_prob += math.log(prob + 1e-10)  # Add small epsilon to avoid log(0)
        
        perplexity = math.exp(-log_prob / total_words)
        
        # Normalize to 0-1 scale (higher = more human-like)
        normalized_perplexity = min(perplexity / 100, 1.0)
        
        return normalized_perplexity
    
    def get_ml_prediction(self, text: str) -> float:
        """Get ML model prediction."""
        try:
            # Vectorize text
            X = self.detector.vectorizer.transform([text])
            # Get prediction probability
            prob = self.detector.classifier.predict_proba(X)[0]
            return prob[1]  # Probability of being AI-generated
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
            perplexity = self.calculate_perplexity_score(text)
            results["perplexity"] = perplexity
        
        if method in ["ML Classification", "Combined Analysis"]:
            ml_score = self.get_ml_prediction(text)
            results["ml_score"] = ml_score
        
        # Combined score calculation
        if method == "Combined Analysis":
            weights = {
                "stylometric": 0.3,
                "perplexity": 0.3,
                "ml": 0.4
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
    st.title("⚙️ Detection Settings")
    detection_method = st.selectbox(
        "Detection Method",
        ["Combined Analysis", "Stylometric Analysis", "Perplexity Analysis", "ML Classification"],
        help="Choose the analysis method. Combined Analysis uses all methods for best accuracy."
    )
    
    show_explanation = st.checkbox("Show Detailed Explanation", value=True)
    show_features = st.checkbox("Show Feature Breakdown", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Analysis Methods")
    st.markdown("""
    **Combined Analysis**: Uses all methods for highest accuracy
    **Stylometric**: Analyzes writing style patterns
    **Perplexity**: Measures text complexity
    **ML Classification**: Machine learning-based detection
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Model Status")
    if os.path.exists("ai_detector_model.pkl"):
        st.success("✅ ML Model Ready")
    else:
        st.info("🔄 Training ML Model...")

# Main content
st.title("🔍 AI Content Detector Pro")
st.markdown("Advanced AI-generated content detection using multiple analysis techniques")

# File upload section
st.markdown("### 📁 Upload Document")
uploaded_file = st.file_uploader(
    "Upload a document (.txt, .docx, .pdf)",
    type=["txt", "docx", "pdf"],
    help="Supported formats: Text files, Word documents, PDF files"
)

# Text input section
st.markdown("### 📝 Or Enter Text Directly")
text_input = st.text_area(
    "Paste your text here",
    height=200,
    placeholder="Enter the text you want to analyze for AI-generated content detection..."
)

# Analysis button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🚀 Analyze Content", use_container_width=True)

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
        with st.spinner("🔍 Analyzing content with advanced algorithms..."):
            results = analyzer.analyze_text(text_content, detection_method)
            
            # Display results
            st.markdown("## 📊 Analysis Results")
            
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
                st.markdown("### 🎯 Confidence Scores")
                
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
                st.markdown("## 🔍 Detailed Analysis")
                
                # Create subplots for detailed analysis
                if detection_method in ["Stylometric Analysis", "Combined Analysis"] and "stylometric" in results:
                    st.markdown("### 📈 Stylometric Analysis")
                    
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
                        st.markdown("#### 📋 Feature Breakdown")
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
                    st.markdown("### 🧮 Perplexity Analysis")
                    
                    perplexity_score = results["perplexity"]
                    
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
                
                if detection_method in ["ML Classification", "Combined Analysis"] and "ml_score" in results:
                    st.markdown("### 🤖 Machine Learning Analysis")
                    
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
            st.markdown("## 💡 Analysis Summary")
            
            if human_probability > 70:
                st.success("✅ **High confidence that this content was written by a human.**")
                st.markdown("The text shows natural variations in writing style, vocabulary usage, and sentence structure typical of human writing.")
            elif human_probability > 50:
                st.warning("⚠️ **Moderate confidence - mixed signals detected.**")
                st.markdown("The analysis shows some characteristics of both human and AI writing. Consider additional context or manual review.")
            else:
                st.error("🚨 **High confidence that this content was AI-generated.**")
                st.markdown("The text exhibits patterns commonly associated with AI-generated content, such as consistent structure and vocabulary usage.")
            
            # Text statistics
            st.markdown("### 📊 Text Statistics")
            word_count = len(text_content.split())
            char_count = len(text_content)
            sentence_count = len(nltk.sent_tokenize(text_content))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Words", f"{word_count:,}")
            with col2:
                st.metric("Characters", f"{char_count:,}")
            with col3:
                st.metric("Sentences", f"{sentence_count:,}")
    
    else:
        st.error("❌ Please provide text or upload a file to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔍 AI Content Detector Pro | Built with ❤️ using Streamlit</p>
    <p>Advanced AI-generated content detection using multiple analysis techniques</p>
</div>
""", unsafe_allow_html=True)