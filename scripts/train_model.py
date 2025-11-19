import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_and_preprocess_data, TextPreprocessor
import warnings
warnings.filterwarnings('ignore')

class KMRLAlertClassifier:
    """KMRL Alert Detection Classifier"""
    
    def __init__(self):
        self.severity_model = None
        self.department_model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.preprocessor = TextPreprocessor()
        
    def train_models(self, df_processed):
        """Train both severity and department classification models"""
        print("Preparing features...")
        
        # Prepare features
        X = df_processed['cleaned_text']
        y_severity = df_processed['severity_encoded']
        y_department = df_processed['department_encoded']
        
        # Split data
        X_train, X_test, y_sev_train, y_sev_test, y_dept_train, y_dept_test = train_test_split(
            X, y_severity, y_department, test_size=0.2, random_state=42, stratify=y_severity
        )
        
        # Vectorize text
        print("Vectorizing text...")
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        # Train severity classifier
        print("Training severity classifier...")
        self.severity_model = LogisticRegression(random_state=42, max_iter=1000)
        self.severity_model.fit(X_train_tfidf, y_sev_train)
        
        # Train department classifier
        print("Training department classifier...")
        self.department_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.department_model.fit(X_train_tfidf, y_dept_train)
        
        # Evaluate models
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Severity predictions
        y_sev_pred = self.severity_model.predict(X_test_tfidf)
        severity_accuracy = accuracy_score(y_sev_test, y_sev_pred)
        
        print(f"\nSEVERITY CLASSIFICATION ACCURACY: {severity_accuracy:.3f}")
        print("\nSeverity Classification Report:")
        severity_classes = self.preprocessor.get_severity_classes()
        print(classification_report(y_sev_test, y_sev_pred, target_names=severity_classes))
        
        # Department predictions
        y_dept_pred = self.department_model.predict(X_test_tfidf)
        department_accuracy = accuracy_score(y_dept_test, y_dept_pred)
        
        print(f"\nDEPARTMENT CLASSIFICATION ACCURACY: {department_accuracy:.3f}")
        print("\nDepartment Classification Report:")
        department_classes = self.preprocessor.get_department_classes()
        print(classification_report(y_dept_test, y_dept_pred, target_names=department_classes))
        
        return X_test, y_sev_test, y_dept_test, y_sev_pred, y_dept_pred
    
    def predict(self, text):
        """Predict severity and department for new text"""
        if isinstance(text, str):
            text = [text]
        
        # Preprocess text
        cleaned_text = [self.preprocessor.clean_text(t) for t in text]
        
        # Vectorize
        text_tfidf = self.tfidf_vectorizer.transform(cleaned_text)
        
        # Predict
        severity_pred = self.severity_model.predict(text_tfidf)
        department_pred = self.department_model.predict(text_tfidf)
        
        # Get probabilities for confidence
        severity_proba = self.severity_model.predict_proba(text_tfidf)
        department_proba = self.department_model.predict_proba(text_tfidf)
        
        # Decode predictions
        severity_labels = self.preprocessor.decode_severity(severity_pred)
        department_labels = self.preprocessor.decode_department(department_pred)
        
        results = []
        for i in range(len(text)):
            # Apply small rule-based booster for safety-critical phrases
            final_severity = severity_labels[i]
            final_department = department_labels[i]
            rule_applied = False
            rule_reason = None

            # Use cleaned text for rule matching
            cleaned = cleaned_text[i]

            # Simple rules: if we see explicit emergency/brake/obstacle phrases, escalate to High
            rules = [
                (r'\bemergency brake\b', ('High', 'Operations', 'Emergency brake detected')),
                (r'\bbrake (?:failure|failed|triggered|malfunction)\b', ('High', 'Operations', 'Brake failure reported')),
                (r'\bobstacle on track\b', ('High', 'Operations', 'Obstacle on track')),
                (r'\bunauthori[sz]ed vendor\b', ('High', 'Safety', 'Unauthorized vendor reported')),
                (r'\bsecurity threat\b', ('High', 'Safety', 'Security threat')),
            ]

            for pattern, (sev_override, dept_override, reason) in rules:
                if re.search(pattern, cleaned):
                    # Only escalate if rule severity is higher than model prediction
                    order = ['Low', 'Medium', 'High', 'Critical']
                    try:
                        if order.index(final_severity) < order.index(sev_override):
                            final_severity = sev_override
                            final_department = dept_override
                            rule_applied = True
                            rule_reason = reason
                    except ValueError:
                        # In case labels don't match expected ordering, apply conservative override
                        final_severity = sev_override
                        final_department = dept_override
                        rule_applied = True
                        rule_reason = reason
                    break

            results.append({
                'text': text[i],
                'severity': final_severity,
                'department': final_department,
                'severity_confidence': max(np.max(severity_proba[i]), 0.6) if rule_applied else np.max(severity_proba[i]),
                'department_confidence': max(np.max(department_proba[i]), 0.6) if rule_applied else np.max(department_proba[i]),
                'alert_required': final_severity in ['Critical', 'High'],
                'rule_applied': rule_applied,
                'rule_reason': rule_reason
            })
        
        return results[0] if len(results) == 1 else results
    
    def save_models(self, model_dir="../models"):
        """Save trained models"""
        joblib.dump(self.severity_model, f"{model_dir}/severity_model.pkl")
        joblib.dump(self.department_model, f"{model_dir}/department_model.pkl")
        joblib.dump(self.tfidf_vectorizer, f"{model_dir}/tfidf_vectorizer.pkl")
        joblib.dump(self.preprocessor, f"{model_dir}/preprocessor.pkl")
        print(f"Models saved to {model_dir}/")
    
    def load_models(self, model_dir="../models"):
        """Load pre-trained models"""
        self.severity_model = joblib.load(f"{model_dir}/severity_model.pkl")
        self.department_model = joblib.load(f"{model_dir}/department_model.pkl")
        self.tfidf_vectorizer = joblib.load(f"{model_dir}/tfidf_vectorizer.pkl")
        self.preprocessor = joblib.load(f"{model_dir}/preprocessor.pkl")
        print(f"Models loaded from {model_dir}/")

def main():
    """Main training pipeline"""
    print("KMRL Alert Detection System - Training Pipeline")
    print("=" * 50)
    
    # Load and preprocess data
    df_processed, preprocessor = load_and_preprocess_data("../data/sample_kmrl_documents.csv")
    
    # Initialize and train classifier
    classifier = KMRLAlertClassifier()
    classifier.preprocessor = preprocessor
    
    # Train models
    test_results = classifier.train_models(df_processed)
    
    # Save models
    classifier.save_models()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("Models saved and ready for deployment.")
    print("="*50)
    
    return classifier

if __name__ == "__main__":
    classifier = main()