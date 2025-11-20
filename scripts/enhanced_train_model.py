import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from enhanced_preprocessing import load_and_preprocess_enhanced_data, AdvancedTextPreprocessor
import warnings
warnings.filterwarnings('ignore')
import re

class EnhancedKMRLClassifier:
    """Enhanced KMRL Alert Detection Classifier with ensemble methods and improved accuracy"""
    
    def __init__(self):
        self.severity_model = None
        self.department_model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased from 5000
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better context
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,  # Use sublinear TF scaling
            norm='l2'
        )
        self.feature_scaler = StandardScaler()
        self.preprocessor = AdvancedTextPreprocessor()
        self.confidence_threshold = 0.7  # Minimum confidence for predictions
        
    def create_ensemble_model(self, model_type='severity'):
        """Create ensemble model with multiple algorithms"""
        if model_type == 'severity':
            # Ensemble for severity classification
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            gb = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
            lr = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                C=1.0
            )
            
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                voting='soft'
            )
        else:
            # Ensemble for department classification
            rf = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                random_state=42,
                class_weight='balanced'
            )
            svm = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            lr = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=0.8
            )
            
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('svm', svm), ('lr', lr)],
                voting='soft'
            )
        
        return ensemble
    
    def train_models(self, df_processed):
        """Train enhanced models with cross-validation and hyperparameter tuning"""
        print("Preparing enhanced features...")
        
        # Prepare text features
        X_text = df_processed['cleaned_text']
        
        # Prepare additional features
        feature_cols = ['emergency_score', 'infrastructure_score', 'safety_score', 'operational_score']
        X_features = df_processed[feature_cols].values
        
        # Prepare targets
        y_severity = df_processed['severity_encoded']
        y_department = df_processed['department_encoded']
        
        # Split data with stratification
        X_text_train, X_text_test, X_feat_train, X_feat_test, y_sev_train, y_sev_test, y_dept_train, y_dept_test = train_test_split(
            X_text, X_features, y_severity, y_department, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_severity
        )
        
        # Vectorize text
        print("Vectorizing text with enhanced TF-IDF...")
        X_text_train_tfidf = self.tfidf_vectorizer.fit_transform(X_text_train)
        X_text_test_tfidf = self.tfidf_vectorizer.transform(X_text_test)
        
        # Scale additional features
        X_feat_train_scaled = self.feature_scaler.fit_transform(X_feat_train)
        X_feat_test_scaled = self.feature_scaler.transform(X_feat_test)
        
        # Combine features
        from scipy.sparse import hstack
        X_train_combined = hstack([X_text_train_tfidf, X_feat_train_scaled])
        X_test_combined = hstack([X_text_test_tfidf, X_feat_test_scaled])
        
        # Train severity classifier with cross-validation
        print("Training enhanced severity classifier...")
        self.severity_model = self.create_ensemble_model('severity')
        
        # Cross-validation for severity
        cv_scores_sev = cross_val_score(
            self.severity_model, X_train_combined, y_sev_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_weighted'
        )
        print(f"Severity CV F1-Score: {cv_scores_sev.mean():.3f} (+/- {cv_scores_sev.std() * 2:.3f})")
        
        self.severity_model.fit(X_train_combined, y_sev_train)
        
        # Train department classifier
        print("Training enhanced department classifier...")
        self.department_model = self.create_ensemble_model('department')
        
        # Cross-validation for department
        cv_scores_dept = cross_val_score(
            self.department_model, X_train_combined, y_dept_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_weighted'
        )
        print(f"Department CV F1-Score: {cv_scores_dept.mean():.3f} (+/- {cv_scores_dept.std() * 2:.3f})")
        
        self.department_model.fit(X_train_combined, y_dept_train)
        
        # Evaluate models
        print("\n" + "="*60)
        print("ENHANCED MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Severity predictions
        y_sev_pred = self.severity_model.predict(X_test_combined)
        y_sev_proba = self.severity_model.predict_proba(X_test_combined)
        
        severity_accuracy = accuracy_score(y_sev_test, y_sev_pred)
        severity_f1 = f1_score(y_sev_test, y_sev_pred, average='weighted')
        
        print(f"\nSEVERITY CLASSIFICATION:")
        print(f"Accuracy: {severity_accuracy:.3f}")
        print(f"F1-Score: {severity_f1:.3f}")
        print(f"CV F1-Score: {cv_scores_sev.mean():.3f}")
        
        severity_classes = self.preprocessor.get_severity_classes()
        print("\nSeverity Classification Report:")
        print(classification_report(y_sev_test, y_sev_pred, target_names=severity_classes))
        
        # Department predictions
        y_dept_pred = self.department_model.predict(X_test_combined)
        y_dept_proba = self.department_model.predict_proba(X_test_combined)
        
        department_accuracy = accuracy_score(y_dept_test, y_dept_pred)
        department_f1 = f1_score(y_dept_test, y_dept_pred, average='weighted')
        
        print(f"\nDEPARTMENT CLASSIFICATION:")
        print(f"Accuracy: {department_accuracy:.3f}")
        print(f"F1-Score: {department_f1:.3f}")
        print(f"CV F1-Score: {cv_scores_dept.mean():.3f}")
        
        department_classes = self.preprocessor.get_department_classes()
        print("\nDepartment Classification Report:")
        print(classification_report(y_dept_test, y_dept_pred, target_names=department_classes))
        
        # Confidence analysis
        sev_confidence = np.max(y_sev_proba, axis=1)
        dept_confidence = np.max(y_dept_proba, axis=1)
        
        print(f"\nCONFIDENCE ANALYSIS:")
        print(f"Average Severity Confidence: {sev_confidence.mean():.3f}")
        print(f"Average Department Confidence: {dept_confidence.mean():.3f}")
        print(f"High Confidence Predictions (>{self.confidence_threshold}): {(sev_confidence > self.confidence_threshold).mean()*100:.1f}%")
        
        return X_test_combined, y_sev_test, y_dept_test, y_sev_pred, y_dept_pred
    
    def predict(self, text):
        """Enhanced prediction with confidence analysis and rule-based boosting"""
        if isinstance(text, str):
            text = [text]
        
        # Preprocess text
        cleaned_text = [self.preprocessor.clean_text(t) for t in text]
        
        # Extract features
        features_list = [self.preprocessor.extract_features(t) for t in text]
        feature_matrix = np.array([[f['emergency_score'], f['infrastructure_score'], 
                                  f['safety_score'], f['operational_score']] for f in features_list])
        
        # Vectorize text
        text_tfidf = self.tfidf_vectorizer.transform(cleaned_text)
        
        # Scale features
        features_scaled = self.feature_scaler.transform(feature_matrix)
        
        # Combine features
        from scipy.sparse import hstack
        X_combined = hstack([text_tfidf, features_scaled])
        
        # Predict
        severity_pred = self.severity_model.predict(X_combined)
        department_pred = self.department_model.predict(X_combined)
        
        # Get probabilities
        severity_proba = self.severity_model.predict_proba(X_combined)
        department_proba = self.department_model.predict_proba(X_combined)
        
        # Decode predictions
        severity_labels = self.preprocessor.decode_severity(severity_pred)
        department_labels = self.preprocessor.decode_department(department_pred)
        
        results = []
        for i in range(len(text)):
            # Apply enhanced rule-based booster
            final_severity = severity_labels[i]
            final_department = department_labels[i]
            rule_applied = False
            rule_reason = None
            confidence_boost = False
            
            cleaned = cleaned_text[i]
            
            # Enhanced rules with more patterns
            rules = [
                (r'\bemergency brake\b', ('Critical', 'Operations', 'Emergency brake activation')),
                (r'\bbomb threat\b', ('Critical', 'Safety', 'Security threat')),
                (r'\bfire detected\b|\bfire broke\b', ('Critical', 'Safety', 'Fire emergency')),
                (r'\btrain (?:collision|derailment)\b', ('Critical', 'Safety', 'Train accident')),
                (r'\bgas leak\b', ('Critical', 'Safety', 'Gas leak emergency')),
                (r'\bstructural damage\b', ('Critical', 'Safety', 'Infrastructure failure')),
                (r'\bobstacle on track\b', ('High', 'Operations', 'Track obstruction')),
                (r'\bbrake (?:failure|failed|malfunction)\b', ('High', 'Operations', 'Brake system failure')),
                (r'\bsignal (?:failure|malfunction)\b', ('High', 'Operations', 'Signal system failure')),
                (r'\bpower (?:failure|outage)\b', ('High', 'Operations', 'Power system failure')),
                (r'\bunauthori[sz]ed (?:personnel|vendor|access)\b', ('High', 'Safety', 'Security breach')),
                (r'\bmedical emergency\b', ('High', 'Safety', 'Medical emergency')),
                (r'\bevacuation\b', ('High', 'Safety', 'Evacuation required')),
            ]
            
            for pattern, (sev_override, dept_override, reason) in rules:
                if re.search(pattern, cleaned):
                    # Check if rule severity is higher
                    severity_order = ['Low', 'Medium', 'High', 'Critical']
                    try:
                        if severity_order.index(final_severity) < severity_order.index(sev_override):
                            final_severity = sev_override
                            final_department = dept_override
                            rule_applied = True
                            rule_reason = reason
                            confidence_boost = True
                    except ValueError:
                        final_severity = sev_override
                        final_department = dept_override
                        rule_applied = True
                        rule_reason = reason
                        confidence_boost = True
                    break
            
            # Calculate confidence
            sev_confidence = np.max(severity_proba[i])
            dept_confidence = np.max(department_proba[i])
            
            # Boost confidence if rule applied or if base confidence is very high
            if confidence_boost:
                sev_confidence = max(sev_confidence, 0.85)
                dept_confidence = max(dept_confidence, 0.8)
            
            # Determine if prediction is reliable
            is_reliable = sev_confidence > self.confidence_threshold and dept_confidence > self.confidence_threshold
            
            results.append({
                'text': text[i],
                'severity': final_severity,
                'department': final_department,
                'severity_confidence': float(sev_confidence),
                'department_confidence': float(dept_confidence),
                'alert_required': final_severity in ['Critical', 'High'],
                'rule_applied': rule_applied,
                'rule_reason': rule_reason,
                'is_reliable': is_reliable,
                'emergency_score': features_list[i]['emergency_score'],
                'safety_score': features_list[i]['safety_score']
            })
        
        return results[0] if len(results) == 1 else results
    
    def save_models(self, model_dir="../models"):
        """Save enhanced models"""
        joblib.dump(self.severity_model, f"{model_dir}/enhanced_severity_model.pkl")
        joblib.dump(self.department_model, f"{model_dir}/enhanced_department_model.pkl")
        joblib.dump(self.tfidf_vectorizer, f"{model_dir}/enhanced_tfidf_vectorizer.pkl")
        joblib.dump(self.feature_scaler, f"{model_dir}/enhanced_feature_scaler.pkl")
        joblib.dump(self.preprocessor, f"{model_dir}/enhanced_preprocessor.pkl")
        print(f"Enhanced models saved to {model_dir}/")
    
    def load_models(self, model_dir="../models"):
        """Load enhanced models"""
        self.severity_model = joblib.load(f"{model_dir}/enhanced_severity_model.pkl")
        self.department_model = joblib.load(f"{model_dir}/enhanced_department_model.pkl")
        self.tfidf_vectorizer = joblib.load(f"{model_dir}/enhanced_tfidf_vectorizer.pkl")
        self.feature_scaler = joblib.load(f"{model_dir}/enhanced_feature_scaler.pkl")
        self.preprocessor = joblib.load(f"{model_dir}/enhanced_preprocessor.pkl")
        print(f"Enhanced models loaded from {model_dir}/")

def main():
    """Main enhanced training pipeline"""
    print("ENHANCED KMRL Alert Detection System - Training Pipeline")
    print("=" * 60)
    
    # Load and preprocess enhanced data
    df_processed, preprocessor = load_and_preprocess_enhanced_data("../data/enhanced_kmrl_documents.csv")
    
    # Initialize and train enhanced classifier
    classifier = EnhancedKMRLClassifier()
    classifier.preprocessor = preprocessor
    
    # Train enhanced models
    test_results = classifier.train_models(df_processed)
    
    # Save enhanced models
    classifier.save_models()
    
    print("\n" + "="*60)
    print("ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
    print("Models trained on 105 diverse documents with ensemble methods")
    print("Enhanced features and rule-based boosting implemented")
    print("Models saved and ready for deployment.")
    print("="*60)
    
    return classifier

if __name__ == "__main__":
    classifier = main()