from flask import Flask, render_template, request, jsonify
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from train_model import KMRLAlertClassifier
import pandas as pd
import json
from datetime import datetime
import time

app = Flask(__name__)

# Global variables
classifier = None
alert_history = []

def load_model():
    """Load the trained model"""
    global classifier
    try:
        classifier = KMRLAlertClassifier()
        classifier.load_models("../models")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def train_model_if_needed():
    """Train model if not available"""
    global classifier
    try:
        from train_model import main
        classifier = main()
        return True
    except Exception as e:
        print(f"Error training model: {e}")
        return False

@app.route('/')
def index():
    """Main dashboard page"""
    # Load sample data for statistics
    try:
        df = pd.read_csv('../data/sample_kmrl_documents.csv')
        stats = {
            'total_docs': len(df),
            'severity_distribution': df['severity'].value_counts().to_dict(),
            'department_distribution': df['department'].value_counts().to_dict(),
            'alert_rate': ((df['severity'] == 'Critical') | (df['severity'] == 'High')).mean() * 100
        }
    except:
        stats = {'total_docs': 0, 'severity_distribution': {}, 'department_distribution': {}, 'alert_rate': 0}
    
    return render_template('index.html', stats=stats, alert_history=alert_history[-10:])

@app.route('/predict', methods=['POST'])
def predict():
    """Process document and return prediction"""
    global classifier, alert_history
    
    # Ensure model is loaded
    if classifier is None:
        if not load_model():
            if not train_model_if_needed():
                return jsonify({'error': 'Failed to load or train model'})
    
    data = request.json
    document_text = data.get('text', '')
    
    if not document_text.strip():
        return jsonify({'error': 'Please enter document text'})
    
    try:
        # Get prediction
        result = classifier.predict(document_text)
        
        # Add timestamp
        result['timestamp'] = datetime.now().strftime('%H:%M:%S')
        result['date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Add to history
        alert_entry = {
            'id': len(alert_history) + 1,
            'text': document_text[:100] + '...' if len(document_text) > 100 else document_text,
            'severity': result['severity'],
            'department': result['department'],
            'alert_required': result['alert_required'],
            'timestamp': result['timestamp'],
            'severity_confidence': result['severity_confidence'],
            'department_confidence': result['department_confidence']
        }
        alert_history.append(alert_entry)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

@app.route('/demo_documents')
def get_demo_documents():
    """Get sample documents for quick testing"""
    demo_docs = [
        "Urgent fire emergency at Aluva station. All passengers evacuated immediately.",
        "Monthly financial report shows steady revenue growth across all stations.", 
        "Signal malfunction causing severe delays. Technical team working on repairs.",
        "New safety training program launched for all operational staff members.",
        "Emergency brake failure reported in Train KM-101. Immediate maintenance required.",
        "Budget allocation approved for station modernization project in Q4 2025.",
        "Power outage at Companyvadi station due to transformer failure detected.",
        "Customer satisfaction survey results show 85% positive feedback received."
    ]
    return jsonify(demo_docs)

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    if not alert_history:
        return render_template('analytics.html', analytics_data={})
    
    # Calculate analytics
    total_processed = len(alert_history)
    alerts_triggered = sum(1 for entry in alert_history if entry['alert_required'])
    alert_rate = (alerts_triggered / total_processed * 100) if total_processed > 0 else 0
    
    # Severity distribution
    severity_counts = {}
    department_counts = {}
    for entry in alert_history:
        sev = entry['severity']
        dept = entry['department']
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        department_counts[dept] = department_counts.get(dept, 0) + 1
    
    analytics_data = {
        'total_processed': total_processed,
        'alerts_triggered': alerts_triggered,
        'alert_rate': round(alert_rate, 1),
        'severity_distribution': severity_counts,
        'department_distribution': department_counts,
        'recent_alerts': [entry for entry in alert_history[-20:] if entry['alert_required']]
    }
    
    return render_template('analytics.html', analytics_data=analytics_data)

@app.route('/comparison')
def comparison():
    """Comparison with traditional approach"""
    return render_template('comparison.html')

if __name__ == '__main__':
    print("🚀 Starting KMRL Alert Detection Web UI...")
    print("Loading model...")
    
    if not load_model():
        print("Model not found. Training new model...")
        if train_model_if_needed():
            print("✅ Model trained successfully!")
        else:
            print("❌ Failed to train model. Some features may not work.")
    else:
        print("✅ Model loaded successfully!")
    
    print("\n🌐 Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)