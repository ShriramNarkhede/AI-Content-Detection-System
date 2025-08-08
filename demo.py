#!/usr/bin/env python3
"""
Demo script for AI Content Detector Pro
Showcases the system with various sample texts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure NLTK data is available
import nltk
try:
    nltk.sent_tokenize("Test sentence.")
except LookupError:
    nltk.download('punkt')

from app import TextAnalyzer
import config

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def print_result(text, results, method):
    """Print formatted results"""
    print(f"\n📝 Sample Text:")
    print(f"   {text[:100]}{'...' if len(text) > 100 else ''}")
    
    print(f"\n🎯 Analysis Results ({method}):")
    
    if method == "Combined Analysis":
        score = results.get("combined_score", 0)
        human_prob = score * 100
        ai_prob = 100 - human_prob
        print(f"   Human-Written Probability: {human_prob:.1f}%")
        print(f"   AI-Generated Probability: {ai_prob:.1f}%")
        
        if human_prob > 70:
            print("   ✅ High confidence: Likely human-written")
        elif human_prob > 50:
            print("   ⚠️  Medium confidence: Mixed signals")
        else:
            print("   🚨 High confidence: Likely AI-generated")
    
    print(f"   Combined Score: {results.get('combined_score', 0):.3f}")
    print(f"   Stylometric Score: {results.get('stylometric_score', 0):.3f}")
    print(f"   Perplexity Score: {results.get('perplexity', 0):.3f}")
    print(f"   ML Score: {results.get('ml_score', 0):.3f}")

def demo_human_texts():
    """Demo with human-like texts"""
    print_header("Human-Like Text Examples")
    
    human_texts = [
        {
            "text": "I went to the store yesterday to buy some groceries. The weather was really nice, so I decided to walk instead of driving. I got bread, milk, eggs, and some vegetables. The cashier was super friendly and we chatted for a bit about the weather.",
            "description": "Casual conversation about daily activities"
        },
        {
            "text": "This book is really hard to put down. The characters are so well-developed and the plot has so many unexpected twists. I stayed up until 2 AM reading it, which was probably not a good idea since I had work the next day.",
            "description": "Personal book review with informal language"
        },
        {
            "text": "I tried cooking that new recipe I found online. It didn't turn out exactly like the picture, but it still tasted pretty good. I think I used too much salt though. My roommate said it was okay, but I could tell she was just being nice.",
            "description": "Personal cooking experience with natural variations"
        }
    ]
    
    analyzer = TextAnalyzer()
    
    for i, sample in enumerate(human_texts, 1):
        print(f"\n--- Example {i}: {sample['description']} ---")
        results = analyzer.analyze_text(sample["text"], "Combined Analysis")
        print_result(sample["text"], results, "Combined Analysis")

def demo_ai_texts():
    """Demo with AI-like texts"""
    print_header("AI-Generated Text Examples")
    
    ai_texts = [
        {
            "text": "The implementation of this system demonstrates a comprehensive approach to data analysis, incorporating multiple methodologies to ensure accurate results. Furthermore, the analysis reveals significant correlations between the variables under investigation, suggesting a robust relationship that warrants further examination.",
            "description": "Academic/technical writing with formal structure"
        },
        {
            "text": "The methodology employed in this study follows established protocols and best practices, ensuring reproducibility and reliability of the experimental outcomes. Additionally, the results indicate a clear pattern of behavior that aligns with theoretical predictions, thereby validating the underlying assumptions of the model.",
            "description": "Research methodology description"
        },
        {
            "text": "The comprehensive review of the literature reveals a consensus among researchers regarding the fundamental principles governing this phenomenon. Moreover, the data analysis demonstrates statistical significance across multiple dimensions, providing strong support for the proposed framework.",
            "description": "Literature review with formal academic language"
        }
    ]
    
    analyzer = TextAnalyzer()
    
    for i, sample in enumerate(ai_texts, 1):
        print(f"\n--- Example {i}: {sample['description']} ---")
        results = analyzer.analyze_text(sample["text"], "Combined Analysis")
        print_result(sample["text"], results, "Combined Analysis")

def demo_analysis_methods():
    """Demo different analysis methods"""
    print_header("Comparison of Analysis Methods")
    
    test_text = """
    The weather today was quite unpredictable. It started sunny but then clouds rolled in 
    and it began to rain. I forgot my umbrella at home, which was pretty annoying. 
    I think I might go back there next week because the prices were pretty good.
    """
    
    analyzer = TextAnalyzer()
    methods = ["Stylometric Analysis", "Perplexity Analysis", "ML Classification", "Combined Analysis"]
    
    print(f"\n📝 Test Text:")
    print(f"   {test_text.strip()}")
    
    for method in methods:
        print(f"\n--- {method} ---")
        results = analyzer.analyze_text(test_text, method)
        
        if method == "Stylometric Analysis":
            score = results.get("stylometric_score", 0)
            human_prob = score * 100
        elif method == "Perplexity Analysis":
            score = results.get("perplexity", 0)
            human_prob = score * 100
        elif method == "ML Classification":
            score = 1 - results.get("ml_score", 0)
            human_prob = score * 100
        else:  # Combined Analysis
            score = results.get("combined_score", 0)
            human_prob = score * 100
        
        ai_prob = 100 - human_prob
        print(f"   Human-Written: {human_prob:.1f}%")
        print(f"   AI-Generated: {ai_prob:.1f}%")
        print(f"   Score: {score:.3f}")

def demo_feature_breakdown():
    """Demo detailed feature breakdown"""
    print_header("Detailed Feature Analysis")
    
    test_text = """
    Machine learning algorithms can process large amounts of data efficiently and identify 
    patterns that might not be immediately apparent to human observers. The implementation 
    demonstrates comprehensive analysis capabilities.
    """
    
    analyzer = TextAnalyzer()
    features = analyzer.calculate_stylometric_features(test_text)
    
    print(f"\n📝 Text: {test_text.strip()}")
    print(f"\n📊 Stylometric Features:")
    
    feature_descriptions = {
        "avg_sentence_length": "Average words per sentence",
        "vocab_richness": "Vocabulary diversity (unique words / total words)",
        "common_words_ratio": "Ratio of frequently used words",
        "sentence_length_variance": "Variation in sentence lengths",
        "avg_word_length": "Average characters per word",
        "word_length_variance": "Variation in word lengths",
        "punctuation_ratio": "Punctuation marks per word",
        "capitalization_ratio": "Capital letters per character",
        "unique_word_ratio": "Unique words per total words"
    }
    
    for feature, value in features.items():
        description = feature_descriptions.get(feature, feature)
        print(f"   {feature}: {value:.3f} ({description})")

def main():
    """Run the complete demo"""
    print("🚀 AI Content Detector Pro - Demo")
    print("="*60)
    print("This demo showcases the AI content detection system with various examples.")
    
    try:
        # Run different demo sections
        demo_human_texts()
        demo_ai_texts()
        demo_analysis_methods()
        demo_feature_breakdown()
        
        print_header("Demo Complete")
        print("✅ All demonstrations completed successfully!")
        print("\n🎯 Key Takeaways:")
        print("   • Human texts show natural variations and informal language")
        print("   • AI texts tend to be more structured and formal")
        print("   • Combined analysis provides the most accurate results")
        print("   • Different methods can give varying confidence levels")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
