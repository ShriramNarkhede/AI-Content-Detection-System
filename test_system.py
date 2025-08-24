#!/usr/bin/env python3
"""
Simple test script to verify the AI Content Detector API is working
"""

import requests
import json

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    # Test text
    test_text = "This is a sample human-written text that demonstrates natural language patterns and variability in sentence structure. It includes various punctuation marks, different sentence lengths, and natural flow that humans typically produce when writing."
    
    print("Testing AI Content Detector API...")
    print(f"Base URL: {base_url}")
    print(f"Test text: {test_text[:50]}...")
    print()
    
    # Test different methods
    methods = [
        "Stylometric Analysis",
        "Perplexity Analysis", 
        "ML Classification",
        "Combined Analysis"
    ]
    
    for method in methods:
        print(f"Testing method: {method}")
        try:
            response = requests.post(
                f"{base_url}/analyze",
                json={"text": test_text, "method": method},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Success - Status: {response.status_code}")
                print(f"  📊 Response keys: {list(data.keys())}")
                
                # Show some key values
                if method == "Stylometric Analysis":
                    if 'stylometric' in data:
                        stylo_data = data['stylometric']
                        if 'stylometric_score' in stylo_data:
                            print(f"  📈 Stylometric score: {stylo_data['stylometric_score']:.3f}")
                
                elif method == "Perplexity Analysis":
                    if 'perplexity' in data:
                        perp_data = data['perplexity']
                        if 'perplexity' in perp_data:
                            print(f"  📈 Perplexity score: {perp_data['perplexity']:.3f}")
                
                elif method == "ML Classification":
                    if 'ml' in data:
                        ml_data = data['ml']
                        if 'ml_score' in ml_data:
                            print(f"  📈 ML score: {ml_data['ml_score']:.3f}")
                
                elif method == "Combined Analysis":
                    if 'combined' in data:
                        combined_data = data['combined']
                        if 'combined_score' in combined_data:
                            print(f"  📈 Combined score: {combined_data['combined_score']:.3f}")
                
                print()
            else:
                print(f"  ❌ Error - Status: {response.status_code}")
                print(f"  📝 Response: {response.text}")
                print()
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Request failed: {e}")
            print()
    
    print("Test completed!")

if __name__ == "__main__":
    test_api()
