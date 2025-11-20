import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedTextPreprocessor:
    """Enhanced text preprocessing pipeline for KMRL documents with better feature extraction"""
    
    def __init__(self):
        self.label_encoder_severity = LabelEncoder()
        self.label_encoder_department = LabelEncoder()
        self.severity_order = ['Low', 'Medium', 'High', 'Critical']
        
    def clean_text(self, text):
        """Enhanced text cleaning with domain-specific preprocessing"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Preserve important domain-specific terms
        text = re.sub(r'\bkm-(\d+)', r'train_km_\1', text)  # Preserve train IDs
        text = re.sub(r'\bplatform\s+(\d+)', r'platform_\1', text)  # Preserve platform numbers
        text = re.sub(r'\bstation\b', 'station_facility', text)  # Enhance station context
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short words but keep important short terms
        words = []
        for word in text.split():
            if len(word) > 1 or word in ['am', 'pm', 'ac', 'it', 'hr']:
                words.append(word)
        
        return ' '.join(words)
    
    def extract_features(self, text):
        """Extract domain-specific features from text"""
        features = {}
        
        # Emergency indicators
        emergency_words = ['emergency', 'urgent', 'immediate', 'critical', 'danger', 'threat', 'evacuation']
        features['emergency_score'] = sum(1 for word in emergency_words if word in text.lower())
        
        # Infrastructure keywords
        infra_words = ['track', 'platform', 'signal', 'brake', 'train', 'station', 'tunnel', 'bridge']
        features['infrastructure_score'] = sum(1 for word in infra_words if word in text.lower())
        
        # Safety keywords
        safety_words = ['fire', 'accident', 'injury', 'medical', 'security', 'evacuation', 'breach']
        features['safety_score'] = sum(1 for word in safety_words if word in text.lower())
        
        # Operational keywords
        ops_words = ['delay', 'maintenance', 'service', 'schedule', 'capacity', 'performance']
        features['operational_score'] = sum(1 for word in ops_words if word in text.lower())
        
        return features
    
    def preprocess_data(self, df):
        """Enhanced preprocessing with feature engineering"""
        df_processed = df.copy()
        
        # Clean text
        df_processed['cleaned_text'] = df_processed['text'].apply(self.clean_text)
        
        # Extract additional features
        feature_data = df_processed['text'].apply(self.extract_features)
        feature_df = pd.DataFrame(feature_data.tolist())
        df_processed = pd.concat([df_processed, feature_df], axis=1)
        
        # Encode labels with proper ordering for severity
        df_processed['severity_encoded'] = self.label_encoder_severity.fit_transform(df_processed['severity'])
        df_processed['department_encoded'] = self.label_encoder_department.fit_transform(df_processed['department'])
        
        return df_processed
    
    def get_severity_classes(self):
        return self.label_encoder_severity.classes_
    
    def get_department_classes(self):
        return self.label_encoder_department.classes_
    
    def decode_severity(self, encoded_labels):
        return self.label_encoder_severity.inverse_transform(encoded_labels)
    
    def decode_department(self, encoded_labels):
        return self.label_encoder_department.inverse_transform(encoded_labels)

def load_and_preprocess_enhanced_data(file_path):
    """Load and preprocess the enhanced KMRL dataset"""
    print("Loading enhanced dataset...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} documents")
    
    # Check data quality
    print("\nData Quality Check:")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate documents: {df.duplicated().sum()}")
    
    # Initialize enhanced preprocessor
    preprocessor = AdvancedTextPreprocessor()
    
    # Preprocess data
    print("Enhanced preprocessing...")
    df_processed = preprocessor.preprocess_data(df)
    
    print("\nDataset Overview:")
    print(f"Severity distribution:\n{df['severity'].value_counts()}")
    print(f"\nDepartment distribution:\n{df['department'].value_counts()}")
    
    # Data balance analysis
    severity_balance = df['severity'].value_counts(normalize=True) * 100
    print(f"\nClass Balance (%):\n{severity_balance.round(1)}")
    
    return df_processed, preprocessor

if __name__ == "__main__":
    # Test the enhanced preprocessing pipeline
    df_processed, preprocessor = load_and_preprocess_enhanced_data("enhanced_kmrl_documents.csv")
    print("\nSample processed data with features:")
    feature_cols = ['emergency_score', 'infrastructure_score', 'safety_score', 'operational_score']
    print(df_processed[['text', 'cleaned_text', 'severity'] + feature_cols].head())