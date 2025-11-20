"""
Test script to debug keyword classification issues
"""
import sys
import os

# Add scripts path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

from scripts.keyword_classifier import KeywordBasedClassifier
from data.alert_keywords import analyze_alert_keywords

def test_keyword_classification():
    """Test various inputs to see classification results"""
    classifier = KeywordBasedClassifier()
    
    test_cases = [
        # Should be LOW
        "passenger information update",
        "train arrival announcement", 
        "normal operations completed",
        "status update for passengers",
        
        # Should be MEDIUM  
        "routine maintenance check",
        "scheduled inspection completed",
        "minor repair needed",
        
        # Should be HIGH
        "signal failure detected", 
        "power disruption reported",
        "unusual noise from engine",
        
        # Should be CRITICAL
        "emergency brake activated",
        "fire detected in workshop",
        "collision risk identified"
    ]
    
    print("🔍 KEYWORD CLASSIFICATION DEBUG TEST")
    print("=" * 60)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{text}'")
        
        # Direct keyword analysis
        result = analyze_alert_keywords(text)
        print(f"   Raw Scores: {result['severity_scores']}")
        print(f"   Context Boost: {result['context_boost']}")
        print(f"   Predicted: {result['severity'].upper()}")
        
        # Classifier analysis  
        classifier_result = classifier.predict(text)
        print(f"   Classifier Result: {classifier_result['severity'].upper()}")
        print(f"   Confidence: {classifier_result['confidence']:.1%}")
        
        # Check if there's a mismatch
        if result['severity'] != classifier_result['severity']:
            print(f"   ⚠️  MISMATCH between direct analysis and classifier!")

if __name__ == "__main__":
    test_keyword_classification()