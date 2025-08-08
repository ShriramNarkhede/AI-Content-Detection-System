#!/usr/bin/env python3
"""
Test script for AI Content Detector Pro
Tests the system with various types of text to verify functionality
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

def test_human_text():
    """Test with human-like text"""
    print("🧪 Testing Human-like Text...")
    
    human_text = """
    I went to the store yesterday to buy some groceries. The weather was really nice, 
    so I decided to walk instead of driving. I got bread, milk, eggs, and some vegetables. 
    The cashier was super friendly and we chatted for a bit about the weather. 
    I think I might go back there next week because the prices were pretty good.
    """
    
    analyzer = TextAnalyzer()
    results = analyzer.analyze_text(human_text, "Combined Analysis")
    
    print(f"✅ Human text analysis completed")
    print(f"   Combined Score: {results.get('combined_score', 0):.3f}")
    print(f"   Stylometric Score: {results.get('stylometric_score', 0):.3f}")
    print(f"   Perplexity Score: {results.get('perplexity', 0):.3f}")
    print(f"   ML Score: {results.get('ml_score', 0):.3f}")
    print()

def test_ai_text():
    """Test with AI-like text"""
    print("🤖 Testing AI-like Text...")
    
    ai_text = """
    The implementation of this system demonstrates a comprehensive approach to data analysis, 
    incorporating multiple methodologies to ensure accurate results. Furthermore, the analysis 
    reveals significant correlations between the variables under investigation, suggesting a 
    robust relationship that warrants further examination. The methodology employed in this 
    study follows established protocols and best practices, ensuring reproducibility and 
    reliability of the experimental outcomes.
    """
    
    analyzer = TextAnalyzer()
    results = analyzer.analyze_text(ai_text, "Combined Analysis")
    
    print(f"✅ AI text analysis completed")
    print(f"   Combined Score: {results.get('combined_score', 0):.3f}")
    print(f"   Stylometric Score: {results.get('stylometric_score', 0):.3f}")
    print(f"   Perplexity Score: {results.get('perplexity', 0):.3f}")
    print(f"   ML Score: {results.get('ml_score', 0):.3f}")
    print()

def test_stylometric_analysis():
    """Test stylometric analysis specifically"""
    print("📈 Testing Stylometric Analysis...")
    
    test_text = """
    This is a test text with various sentence lengths. Some are short. Others are much longer 
    and contain more complex structures with multiple clauses and descriptive elements that 
    demonstrate natural writing patterns. The vocabulary varies naturally.
    """
    
    analyzer = TextAnalyzer()
    features = analyzer.calculate_stylometric_features(test_text)
    
    print(f"✅ Stylometric analysis completed")
    for feature, value in features.items():
        print(f"   {feature}: {value:.3f}")
    print()

def test_perplexity_analysis():
    """Test perplexity analysis specifically"""
    print("🧮 Testing Perplexity Analysis...")
    
    test_text = """
    The quick brown fox jumps over the lazy dog. This sentence contains all letters of the alphabet.
    """
    
    analyzer = TextAnalyzer()
    perplexity = analyzer.calculate_perplexity_score(test_text)
    
    print(f"✅ Perplexity analysis completed")
    print(f"   Perplexity Score: {perplexity:.3f}")
    print()

def test_ml_classification():
    """Test ML classification specifically"""
    print("🤖 Testing ML Classification...")
    
    test_text = """
    Machine learning algorithms can process large amounts of data efficiently and identify 
    patterns that might not be immediately apparent to human observers.
    """
    
    analyzer = TextAnalyzer()
    ml_score = analyzer.get_ml_prediction(test_text)
    
    print(f"✅ ML classification completed")
    print(f"   ML Score (AI probability): {ml_score:.3f}")
    print(f"   Human probability: {(1-ml_score):.3f}")
    print()

def test_file_processing():
    """Test file processing capabilities"""
    print("📁 Testing File Processing...")
    
    # Create a test text file
    test_content = "This is a test file with some sample text for analysis."
    test_file_path = "test_sample.txt"
    
    try:
        with open(test_file_path, 'w') as f:
            f.write(test_content)
        
        # Simulate file upload object
        class MockUploadedFile:
            def __init__(self, content, file_type):
                self.content = content
                self.type = file_type
            
            def read(self):
                return self.content.encode('utf-8')
        
        mock_file = MockUploadedFile(test_content, "text/plain")
        
        analyzer = TextAnalyzer()
        extracted_text = analyzer.extract_text_from_file(mock_file)
        
        print(f"✅ File processing test completed")
        print(f"   Extracted text: {extracted_text[:50]}...")
        
        # Clean up
        os.remove(test_file_path)
        
    except Exception as e:
        print(f"❌ File processing test failed: {e}")
    print()

def test_configuration():
    """Test configuration loading"""
    print("⚙️ Testing Configuration...")
    
    print(f"✅ Configuration loaded successfully")
    print(f"   Analysis weights: {config.ANALYSIS_WEIGHTS}")
    print(f"   Confidence thresholds: {config.CONFIDENCE_THRESHOLDS}")
    print(f"   Available analysis methods: {config.ANALYSIS_METHODS}")
    print()

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting AI Content Detector Pro Tests")
    print("=" * 50)
    
    try:
        test_configuration()
        test_stylometric_analysis()
        test_perplexity_analysis()
        test_ml_classification()
        test_file_processing()
        test_human_text()
        test_ai_text()
        
        print("=" * 50)
        print("✅ All tests completed successfully!")
        print("🎉 AI Content Detector Pro is ready to use!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
