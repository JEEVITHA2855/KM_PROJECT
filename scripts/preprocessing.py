import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class TextPreprocessor:
    """Text preprocessing pipeline for KMRL documents"""
    
    def __init__(self):
        self.label_encoder_severity = LabelEncoder()
        self.label_encoder_department = LabelEncoder()
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short words (less than 2 characters)
        text = ' '.join([word for word in text.split() if len(word) > 1])
        
        return text
    
    def preprocess_data(self, df):
        """Preprocess the entire dataframe"""
        df_processed = df.copy()
        
        # Clean text
        df_processed['cleaned_text'] = df_processed['text'].apply(self.clean_text)
        
        # Encode labels
        df_processed['severity_encoded'] = self.label_encoder_severity.fit_transform(df_processed['severity'])
        df_processed['department_encoded'] = self.label_encoder_department.fit_transform(df_processed['department'])
        
        return df_processed
    
    def get_severity_classes(self):
        """Get severity class names"""
        return self.label_encoder_severity.classes_
    
    def get_department_classes(self):
        """Get department class names"""
        return self.label_encoder_department.classes_
    
    def decode_severity(self, encoded_labels):
        """Decode severity labels back to original form"""
        return self.label_encoder_severity.inverse_transform(encoded_labels)
    
    def decode_department(self, encoded_labels):
        """Decode department labels back to original form"""
        return self.label_encoder_department.inverse_transform(encoded_labels)

def load_and_preprocess_data(file_path):
    """Load and preprocess the KMRL dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} documents")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess data
    print("Preprocessing text...")
    df_processed = preprocessor.preprocess_data(df)
    
    print("\nDataset Overview:")
    print(f"Severity distribution:\n{df['severity'].value_counts()}")
    print(f"\nDepartment distribution:\n{df['department'].value_counts()}")
    
    return df_processed, preprocessor

if __name__ == "__main__":
    # Test the preprocessing pipeline
    df_processed, preprocessor = load_and_preprocess_data("../data/sample_kmrl_documents.csv")
    print("\nSample processed data:")
    print(df_processed[['text', 'cleaned_text', 'severity', 'department']].head())