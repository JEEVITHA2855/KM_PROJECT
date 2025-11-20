# KMRL Alert Detection System - Enhanced Production Deployment

## Overview
This enhanced KMRL alert detection system uses ensemble machine learning models to classify document severity (Critical, High, Medium, Low) and department (Operations, Maintenance, Safety). The system achieves 57.1% accuracy for severity and 71.4% for department classification with confidence scoring and rule-based safety boosters.

## Model Performance
- **Dataset**: 105 diverse KMRL documents
- **Severity Classification**: 57.1% accuracy, F1-Score: 0.552
- **Department Classification**: 71.4% accuracy, F1-Score: 0.695
- **High Confidence Predictions**: 19.0% (above 0.7 threshold)
- **Cross-Validation**: Robust 5-fold stratified validation

## Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/JEEVITHA2855/KM_PROJECT.git
cd KM_PROJECT

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Training (Optional - Pre-trained models included)
```bash
# Train enhanced models with 105 documents
python scripts/enhanced_train_model.py
```

### 3. Web Application
```bash
# Start Flask server
cd web_app
python app.py
```
Visit: http://localhost:5000

## Features

### Enhanced Machine Learning
- **Ensemble Methods**: VotingClassifier with RandomForest, GradientBoosting, LogisticRegression, SVM
- **Advanced Preprocessing**: Domain-specific feature extraction with emergency, infrastructure, safety, and operational scores
- **Feature Engineering**: TF-IDF with trigrams, 10,000 features, sublinear scaling
- **Confidence Scoring**: 0.7 minimum threshold with reliability indicators

### Safety Boosters
Rule-based escalation for critical phrases:
- "emergency brake", "derailment", "collision"
- "brake failure", "signal failure", "power failure"
- "obstacle on track", "unauthorized entry"
- "fire", "smoke", "evacuation"

### User Interface
- **Professional Design**: Clean, icon-free interface
- **Real-time Analysis**: Instant classification with confidence bars
- **Reliability Indicators**: Visual feedback on prediction quality
- **Alert History**: Track and review past classifications
- **Comparison Tool**: Side-by-side document analysis

## File Structure
```
KM_PROJECT/
├── data/
│   ├── enhanced_kmrl_documents.csv     # 105 training documents
│   └── labeling_guidelines.md          # Classification guidelines
├── models/
│   ├── enhanced_severity_model.joblib  # Ensemble severity classifier
│   ├── enhanced_dept_model.joblib      # Ensemble department classifier
│   └── enhanced_preprocessor.joblib    # Advanced text preprocessor
├── scripts/
│   ├── enhanced_preprocessing.py       # AdvancedTextPreprocessor
│   ├── enhanced_train_model.py         # EnhancedKMRLClassifier
│   └── demo.py                         # Command-line demo
├── web_app/
│   ├── app.py                          # Flask application
│   └── templates/                      # HTML templates
└── notebooks/
    └── KMRL_Alert_Detection_Demo.ipynb # Interactive demo
```

## API Usage

### Command Line
```python
from scripts.enhanced_train_model import EnhancedKMRLClassifier

# Load model
classifier = EnhancedKMRLClassifier()
classifier.load_models('../models')

# Predict
result = classifier.predict("Emergency brake triggered in Train KMRL-108")
print(f"Severity: {result['severity']}, Department: {result['department']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Web API
```python
import requests

response = requests.post('http://localhost:5000/predict', 
                        json={'text': 'Signal failure at Bangalore East'})
result = response.json()
```

## Production Considerations

### Performance Optimization
- Models load with fallback logic (enhanced → basic)
- Preprocessing cached for repeated predictions
- Cross-validation ensures robust performance

### Monitoring
- Confidence thresholds alert to uncertain predictions
- Rule-based boosters provide explainable safety escalation
- Alert history enables performance tracking

### Scalability
- Stateless design supports horizontal scaling
- Models serialized with joblib for fast loading
- Ensemble approach balances accuracy and speed

## Continuous Learning
1. **Data Collection**: Save new predictions for retraining
2. **Model Updates**: Retrain monthly with new documents
3. **Performance Monitoring**: Track accuracy on validation set
4. **Rule Updates**: Adapt safety boosters based on incidents

## Troubleshooting

### Model Not Loading
- Check models/ directory contains enhanced_*.joblib files
- Fallback to basic models if enhanced versions unavailable
- Verify joblib version compatibility

### Low Accuracy
- Check document format matches training data
- Ensure text preprocessing steps are consistent
- Consider confidence threshold adjustment

### Web App Issues
- Verify Flask installation: `pip install flask`
- Check port 5000 availability
- Review console logs for detailed errors

## Support
For technical support or feature requests, please contact the development team or create an issue on GitHub.