"""
Configuration file for AI Content Detector Pro
Contains all configurable parameters and settings
"""

# Analysis Method Weights (for Combined Analysis)
ANALYSIS_WEIGHTS = {
    "stylometric": 0.3,
    "perplexity": 0.3,
    "ml": 0.4
}

# Confidence Level Thresholds
CONFIDENCE_THRESHOLDS = {
    "high": 70,
    "medium": 50,
    "low": 0
}

# Stylometric Feature Weights
STYLOMETRIC_WEIGHTS = {
    "common_words_ratio": 0.3,
    "unique_word_ratio": 0.3,
    "punctuation_ratio": 0.2,
    "sentence_length_variance": 0.2
}

# ML Model Configuration
ML_CONFIG = {
    "max_features": 1000,
    "n_estimators": 100,
    "random_state": 42,
    "model_path": "ai_detector_model.pkl"
}

# Text Processing Settings
TEXT_PROCESSING = {
    "min_text_length": 10,
    "max_text_length": 50000,
    "perplexity_normalization": 100
}

# UI Configuration
UI_CONFIG = {
    "page_title": "AI Content Detector Pro",
    "page_icon": "🔍",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Color Scheme
COLORS = {
    "human": "#00CC96",
    "ai": "#EF553B",
    "medium": "#FFA500",
    "primary": "#636EFA",
    "background": "#f0f2f6",
    "highlight": "#e8f4fd"
}

# File Upload Settings
FILE_TYPES = {
    "allowed_types": ["txt", "docx", "pdf"],
    "max_size_mb": 10
}

# Analysis Methods
ANALYSIS_METHODS = [
    "Combined Analysis",
    "Stylometric Analysis", 
    "Perplexity Analysis",
    "ML Classification"
]

# Human-like Text Patterns (for training data generation)
HUMAN_TEXT_PATTERNS = [
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

# AI-like Text Patterns (for training data generation)
AI_TEXT_PATTERNS = [
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

# Training Data Configuration
TRAINING_CONFIG = {
    "human_samples": 100,
    "ai_samples": 100,
    "human_variations": [
        "Anyway, that's what happened.",
        "I'm not really sure about that.",
        "What do you think?",
        "It was kind of weird.",
        "I guess we'll see."
    ],
    "ai_variations": [
        "This conclusion is supported by extensive empirical evidence.",
        "Further research is warranted to explore these findings.",
        "The implications of these results are far-reaching.",
        "This analysis provides valuable insights for practitioners.",
        "The methodology ensures robust and reliable outcomes."
    ]
}

# Error Messages
ERROR_MESSAGES = {
    "no_text": "Please provide text to analyze.",
    "file_read_error": "Could not extract text from the uploaded file.",
    "ml_prediction_failed": "ML prediction failed: {}",
    "model_loading_failed": "Model loading failed, training new model...",
    "text_too_short": "Text is too short for meaningful analysis.",
    "text_too_long": "Text is too long for analysis."
}

# Success Messages
SUCCESS_MESSAGES = {
    "model_loaded": "✅ Model loaded successfully",
    "model_trained": "✅ Model trained and saved",
    "analysis_complete": "Analysis completed successfully"
}

# Info Messages
INFO_MESSAGES = {
    "training_model": "🔄 Training new model...",
    "model_ready": "✅ ML Model Ready",
    "training_in_progress": "🔄 Training ML Model..."
}
