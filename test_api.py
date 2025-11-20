"""
Test the Flask API directly to see what's being returned
"""
import requests
import json

def test_flask_api():
    """Test the Flask predict endpoint directly"""
    url = "http://localhost:5000/predict"
    
    test_cases = [
        "passenger information update",
        "routine maintenance check", 
        "signal failure detected",
        "emergency brake activated"
    ]
    
    print("🌐 FLASK API TEST")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        try:
            response = requests.post(url, json={"text": text})
            data = response.json()
            
            print(f"\n{i}. Input: '{text}'")
            print(f"   Severity: {data.get('severity', 'ERROR')}")
            print(f"   Department: {data.get('department', 'ERROR')}")
            print(f"   Confidence: {data.get('confidence', 0):.1%}")
            print(f"   Matched Keywords: {data.get('matched_keywords', {})}")
            
        except Exception as e:
            print(f"\n{i}. Input: '{text}'")
            print(f"   ERROR: {str(e)}")

if __name__ == "__main__":
    test_flask_api()