"""
Keyword-based KMRL Alert Detection System
Uses relevant stopwords and keyword matching for classification
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.alert_keywords import (
    analyze_alert_keywords, 
    SEVERITY_KEYWORDS, 
    DEPARTMENT_KEYWORDS,
    CONTEXT_KEYWORDS
)

class KeywordBasedClassifier:
    """
    Keyword-based classifier that uses relevant stopwords
    to predict severity and department based on word matching
    """
    
    def __init__(self):
        self.severity_keywords = SEVERITY_KEYWORDS
        self.department_keywords = DEPARTMENT_KEYWORDS
        self.context_keywords = CONTEXT_KEYWORDS
        
    def predict(self, text):
        """
        Predict severity and department based on keyword matching
        """
        if not text or not text.strip():
            return {
                "severity": "low",
                "department": "operations", 
                "confidence": 0.0,
                "error": "Empty text provided"
            }
        
        # Analyze using keywords
        result = analyze_alert_keywords(text)
        
        # Extract matched keywords for explanation
        matched_keywords = self._get_matched_keywords(text)
        
        return {
            "severity": result["severity"],
            "department": result["department"],
            "confidence": result["overall_confidence"],
            "severity_confidence": result["severity_confidence"],
            "department_confidence": result["department_confidence"],
            "severity_scores": result["severity_scores"],
            "department_scores": result["department_scores"],
            "context_boost": result["context_boost"],
            "matched_keywords": matched_keywords,
            "explanation": self._generate_explanation(text, result, matched_keywords)
        }
    
    def _get_matched_keywords(self, text):
        """Extract which specific keywords were matched"""
        text_lower = text.lower()
        matched = {
            "severity": {},
            "department": {},
            "context": {}
        }
        
        # Find severity keyword matches
        for severity, data in self.severity_keywords.items():
            matches = [kw for kw in data["keywords"] if kw in text_lower]
            if matches:
                matched["severity"][severity] = matches
        
        # Find department keyword matches  
        for dept, data in self.department_keywords.items():
            matches = [kw for kw in data["keywords"] if kw in text_lower]
            if matches:
                matched["department"][dept] = matches
                
        # Find context keyword matches
        for context_type, keywords in self.context_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                matched["context"][context_type] = matches
        
        return matched
    
    def _generate_explanation(self, text, result, matched_keywords):
        """Generate human-readable explanation"""
        explanation = []
        
        # Severity explanation
        severity_matches = matched_keywords.get("severity", {})
        if severity_matches:
            for sev, keywords in severity_matches.items():
                if keywords:
                    explanation.append(f"'{sev.title()}' severity keywords found: {', '.join(keywords)}")
        
        # Department explanation
        dept_matches = matched_keywords.get("department", {})
        if dept_matches:
            for dept, keywords in dept_matches.items():
                if keywords:
                    explanation.append(f"'{dept.title()}' department keywords found: {', '.join(keywords)}")
        
        # Context explanation
        context_matches = matched_keywords.get("context", {})
        if context_matches:
            for ctx, keywords in context_matches.items():
                if keywords:
                    explanation.append(f"Railway context ({ctx}) keywords: {', '.join(keywords)}")
        
        # Overall prediction
        explanation.append(f"Predicted: {result['severity'].title()} severity, {result['department'].title()} department")
        explanation.append(f"Confidence: {result['overall_confidence']:.1%}")
        
        return " | ".join(explanation)
    
    def get_keyword_summary(self):
        """Get summary of all keywords used"""
        summary = {
            "severity_keywords": {k: len(v["keywords"]) for k, v in self.severity_keywords.items()},
            "department_keywords": {k: len(v["keywords"]) for k, v in self.department_keywords.items()},
            "context_keywords": {k: len(v) for k, v in self.context_keywords.items()},
            "total_keywords": (
                sum(len(v["keywords"]) for v in self.severity_keywords.values()) +
                sum(len(v["keywords"]) for v in self.department_keywords.values()) +
                sum(len(v) for v in self.context_keywords.values())
            )
        }
        return summary

def main():
    """Demo the keyword-based classifier"""
    classifier = KeywordBasedClassifier()
    
    print("🔍 KMRL Keyword-Based Alert Detection System")
    print("=" * 50)
    
    # Show keyword summary
    summary = classifier.get_keyword_summary()
    print(f"\n📊 Keyword Summary:")
    print(f"• Severity keywords: {summary['severity_keywords']}")
    print(f"• Department keywords: {summary['department_keywords']}")  
    print(f"• Context keywords: {summary['context_keywords']}")
    print(f"• Total keywords: {summary['total_keywords']}")
    
    # Test examples
    test_cases = [
        "Emergency brake triggered in Train KMRL-108 due to obstacle on track",
        "Routine maintenance check completed on overhead contact system",
        "Signal failure detected at Kaloor station, delays expected", 
        "Passenger announcement for train departure from Aluva",
        "Fire detected in depot workshop, evacuation in progress",
        "Minor noise reported from wheel assembly during inspection"
    ]
    
    print(f"\n🧪 Test Cases:")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        result = classifier.predict(text)
        
        print(f"\n{i}. Text: {text}")
        print(f"   Severity: {result['severity'].upper()} ({result['severity_confidence']:.1%})")
        print(f"   Department: {result['department'].upper()} ({result['department_confidence']:.1%})")
        print(f"   Keywords: {result['matched_keywords']}")
        print(f"   Explanation: {result['explanation']}")

if __name__ == "__main__":
    main()