from flask import Flask, render_template, request, jsonify
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Import keyword-based classifier
from keyword_classifier import KeywordBasedClassifier
    
import pandas as pd
import json
from datetime import datetime
import time

app = Flask(__name__)

# Global classifier instance
classifier = None

def load_classifier():
    """Load the keyword-based classifier"""
    global classifier
    try:
        print("Loading keyword-based classifier...")
        classifier = KeywordBasedClassifier()
        print("✅ Keyword-based classifier loaded successfully!")
        
        # Show keyword summary
        summary = classifier.get_keyword_summary()
        print(f"📊 Total keywords loaded: {summary['total_keywords']}")
        print(f"   - Severity: {sum(summary['severity_keywords'].values())}")
        print(f"   - Department: {sum(summary['department_keywords'].values())}")
        print(f"   - Context: {sum(summary['context_keywords'].values())}")
        
        return True
            
    except Exception as e:
        print(f"Error loading classifier: {str(e)}")
        return False

# Alert history storage
alert_history = []

def store_alert_history(prediction_result):
    """Store prediction in alert history"""
    global alert_history
    alert_history.append(prediction_result)
    
    # Keep only last 50 alerts
    if len(alert_history) > 50:
        alert_history = alert_history[-50:]

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/comparison')
def comparison():
    """Comparison page for analyzing multiple documents"""
    return render_template('comparison.html')

@app.route('/analytics')
def analytics():
    """Analytics page showing alert history and trends"""
    return render_template('analytics.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'error': 'No text provided'
            }), 400
        
        if not classifier:
            return jsonify({
                'error': 'Classifier not loaded'
            }), 500
        
        # Get prediction from keyword classifier
        result = classifier.predict(text)
        
        # Format response
        response = {
            'severity': result.get('severity', 'Unknown'),
            'department': result.get('department', 'Unknown'), 
            'confidence': float(result.get('confidence', 0)),
            'severity_confidence': float(result.get('severity_confidence', 0)),
            'department_confidence': float(result.get('department_confidence', 0)),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'text': text,
            'model_type': 'keyword_based',
            'matched_keywords': result.get('matched_keywords', {}),
            'explanation': result.get('explanation', ''),
            'context_boost': float(result.get('context_boost', 0)),
            'severity_scores': result.get('severity_scores', {}),
            'department_scores': result.get('department_scores', {})
        }
        
        # Store in alert history
        store_alert_history(response)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/compare', methods=['POST'])
def compare():
    """Compare multiple documents"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({
                'error': 'No texts provided for comparison'
            }), 400
        
        if not classifier:
            return jsonify({
                'error': 'Classifier not loaded'
            }), 500
        
        results = []
        for i, text in enumerate(texts):
            if text.strip():
                result = classifier.predict(text.strip())
                result['document_id'] = f"Document {i+1}"
                result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                result['model_type'] = 'keyword_based'
                results.append(result)
        
        return jsonify({
            'results': results,
            'comparison_summary': generate_comparison_summary(results)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Comparison failed: {str(e)}'
        }), 500

@app.route('/history')
def get_history():
    """Get alert history"""
    try:
        return jsonify({
            'history': alert_history,
            'total_alerts': len(alert_history)
        })
    except Exception as e:
        return jsonify({
            'error': f'Failed to retrieve history: {str(e)}'
        }), 500

@app.route('/keywords')
def get_keywords():
    """Get keyword information"""
    try:
        if not classifier:
            return jsonify({'error': 'Classifier not loaded'}), 500
        
        summary = classifier.get_keyword_summary()
        return jsonify({
            'summary': summary,
            'severity_keywords': classifier.severity_keywords,
            'department_keywords': classifier.department_keywords,
            'context_keywords': classifier.context_keywords
        })
    except Exception as e:
        return jsonify({
            'error': f'Failed to retrieve keywords: {str(e)}'
        }), 500

def generate_comparison_summary(results):
    """Generate summary for comparison results"""
    if not results:
        return {}
    
    # Count severity levels
    severity_counts = {}
    department_counts = {}
    
    for result in results:
        severity = result.get('severity', 'Unknown')
        department = result.get('department', 'Unknown')
        
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        department_counts[department] = department_counts.get(department, 0) + 1
    
    # Find highest priority
    severity_priority = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
    highest_severity = max(results, key=lambda x: severity_priority.get(x.get('severity', 'low'), 1))
    
    return {
        'total_documents': len(results),
        'severity_distribution': severity_counts,
        'department_distribution': department_counts,
        'highest_priority': {
            'severity': highest_severity.get('severity'),
            'department': highest_severity.get('department'),
            'document_id': highest_severity.get('document_id')
        },
        'avg_confidence': sum(r.get('confidence', 0) for r in results) / len(results)
    }

if __name__ == '__main__':
    print("🚀 Starting KMRL Keyword-Based Alert Detection Web UI...")
    
    # Load classifier at startup
    if load_classifier():
        print("✅ Classifier loaded successfully!")
        
        print("\n🌐 Open your browser and go to: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load keyword classifier. Please check your installation.")
                print("Enhanced models not found, trying basic models...")
                # Try loading basic models with enhanced classifier
                try:
                    # Load basic models but use enhanced classifier structure
                    from train_model import KMRLAlertClassifier as BasicClassifier
                    basic_classifier = BasicClassifier()
                    basic_classifier.load_models("../models")
                    classifier = basic_classifier
                    print(f"✅ Basic model loaded with enhanced classifier!")
                    return True
                except:
                    return False
        else:
            classifier.load_models("../models")
            print(f"✅ Basic model loaded successfully!")
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
    # Load sample data for statistics (try enhanced dataset first)
    try:
        df = pd.read_csv('../data/enhanced_kmrl_documents.csv')
        dataset_type = "Enhanced"
    except:
        try:
            df = pd.read_csv('../data/sample_kmrl_documents.csv')
            dataset_type = "Basic"
        except:
            df = None
            dataset_type = "None"
    
    if df is not None:
        stats = {
            'total_docs': len(df),
            'severity_distribution': df['severity'].value_counts().to_dict(),
            'department_distribution': df['department'].value_counts().to_dict(),
            'alert_rate': ((df['severity'] == 'Critical') | (df['severity'] == 'High')).mean() * 100,
            'dataset_type': dataset_type,
            'model_type': MODEL_TYPE
        }
    else:
        stats = {
            'total_docs': 0, 
            'severity_distribution': {}, 
            'department_distribution': {}, 
            'alert_rate': 0,
            'dataset_type': dataset_type,
            'model_type': MODEL_TYPE
        }
    
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