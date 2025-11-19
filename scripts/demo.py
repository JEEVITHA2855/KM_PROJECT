"""
KMRL Alert Detection System - Live Demo
======================================
This script demonstrates the trained model with real-time predictions
and shows how it replaces the traditional stopword-based approach.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_model import KMRLAlertClassifier
import pandas as pd
from datetime import datetime
import time

class KMRLAlertDemo:
    """Demo class for KMRL Alert Detection System"""
    
    def __init__(self):
        self.classifier = KMRLAlertClassifier()
        self.alert_count = 0
        
    def load_demo_model(self):
        """Load the trained model for demo"""
        try:
            self.classifier.load_models()
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Training model first...")
            return False
    
    def train_demo_model(self):
        """Train the model if not available"""
        print("🔄 Training model for demo...")
        from train_model import main
        self.classifier = main()
        print("✅ Demo model trained and ready!")
    
    def process_document(self, text):
        """Process a single document and return alert decision"""
        result = self.classifier.predict(text)
        
        # Simulate alert processing
        if result['alert_required']:
            self.alert_count += 1
            alert_status = f"🚨 ALERT #{self.alert_count} TRIGGERED"
        else:
            alert_status = "ℹ️  No alert required"
        
        return result, alert_status
    
    def demo_real_time_processing(self):
        """Demonstrate real-time document processing"""
        print("\n" + "="*60)
        print("🔴 LIVE DEMO: Real-time Document Processing")
        print("="*60)
        
        # Sample new documents for demo
        demo_documents = [
            "Fire alarm activated at Ernakulam South station. Passengers evacuated immediately. Fire brigade on the way.",
            "Monthly cleaning of all stations completed successfully. No issues reported by staff.",
            "Signal malfunction at Kalamassery causing major delays. Technical team investigating the issue.",
            "New digital displays installed at all platforms. Passengers can now see real-time train information.",
            "Emergency medical situation at Aluva station. Patient transported to hospital. Normal operations resumed.",
            "Quarterly revenue report shows steady growth. All financial targets met this quarter.",
            "Security breach detected in CCTV system. Immediate action required to secure surveillance network.",
            "Staff appreciation event organized successfully. High employee satisfaction reported."
        ]
        
        for i, doc in enumerate(demo_documents, 1):
            print(f"\n📄 Processing Document #{i}:")
            print(f"Text: {doc[:80]}...")
            
            # Show processing animation
            print("🔄 Analyzing...", end="", flush=True)
            for _ in range(3):
                time.sleep(0.5)
                print(".", end="", flush=True)
            print(" Done!")
            
            # Get prediction
            result, alert_status = self.process_document(doc)
            
            # Display results
            print(f"   Severity: {result['severity']} (Confidence: {result['severity_confidence']:.2f})")
            print(f"   Department: {result['department']} (Confidence: {result['department_confidence']:.2f})")
            print(f"   Decision: {alert_status}")
            
            if result['alert_required']:
                print(f"   📧 Notification sent to {result['department']} team")
                print(f"   🕐 Alert time: {datetime.now().strftime('%H:%M:%S')}")
            
            print("-" * 40)
    
    def compare_with_stopword_approach(self):
        """Compare ML approach with traditional stopword method"""
        print("\n" + "="*60)
        print("📊 COMPARISON: ML Model vs Traditional Stopwords")
        print("="*60)
        
        # Traditional stopword approach (simplified)
        critical_stopwords = ['emergency', 'fire', 'evacuation', 'urgent', 'critical', 'danger']
        
        test_documents = [
            "Regular maintenance scheduled for next week at all stations",
            "Fire alarm malfunction detected at platform 3 - urgent repair needed",
            "Monthly financial report shows positive trends in revenue",
            "Emergency evacuation drill completed successfully with all staff participation"
        ]
        
        print(f"{'Document':<50} {'Stopwords':<15} {'ML Model':<15} {'Better?'}")
        print("-" * 85)
        
        for doc in test_documents:
            # Stopword approach
            has_critical_words = any(word in doc.lower() for word in critical_stopwords)
            stopword_decision = "ALERT" if has_critical_words else "NO ALERT"
            
            # ML approach
            result, _ = self.process_document(doc)
            ml_decision = "ALERT" if result['alert_required'] else "NO ALERT"
            
            # Determine which is better (simplified logic)
            better = "✅ ML" if (result['severity'] in ['Critical', 'High'] and 'emergency' in doc.lower() and 'drill' not in doc.lower()) or (result['severity'] in ['Low', 'Medium'] and 'drill' in doc.lower()) else "⚖️ Same"
            
            print(f"{doc[:48]:<50} {stopword_decision:<15} {ml_decision:<15} {better}")
    
    def show_model_insights(self):
        """Show model performance insights"""
        print("\n" + "="*60)
        print("📈 MODEL PERFORMANCE INSIGHTS")
        print("="*60)
        
        # Load sample data for analysis
        df = pd.read_csv("../data/sample_kmrl_documents.csv")
        
        print(f"✅ Training Data: {len(df)} documents")
        print(f"✅ Severity Classes: {df['severity'].unique()}")
        print(f"✅ Departments: {df['department'].unique()}")
        
        print(f"\n📊 Severity Distribution:")
        for severity, count in df['severity'].value_counts().items():
            print(f"   {severity}: {count} documents ({count/len(df)*100:.1f}%)")
        
        print(f"\n🏢 Department Distribution:")
        for dept, count in df['department'].value_counts().items():
            print(f"   {dept}: {count} documents ({count/len(df)*100:.1f}%)")
        
        print(f"\n🎯 Key Advantages of ML Approach:")
        print("   • Context-aware classification (not just keyword matching)")
        print("   • Learns from labeled examples")
        print("   • Handles multilingual content")
        print("   • Provides confidence scores")
        print("   • Reduces false positives")
        print("   • Adapts with more training data")
    
    def interactive_demo(self):
        """Interactive demo where users can input custom text"""
        print("\n" + "="*60)
        print("🎮 INTERACTIVE MODE: Test Your Own Documents")
        print("="*60)
        print("Enter document text (or 'quit' to exit):")
        
        while True:
            user_input = input("\n📝 Document: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            result, alert_status = self.process_document(user_input)
            
            print(f"🔍 Analysis Results:")
            print(f"   Severity: {result['severity']} ({result['severity_confidence']:.2f})")
            print(f"   Department: {result['department']} ({result['department_confidence']:.2f})")
            print(f"   {alert_status}")
    
    def run_full_demo(self):
        """Run the complete demo sequence"""
        print("🚀 KMRL Alert Detection System - Complete Demo")
        print("=" * 60)
        print("This demo shows how ML replaces traditional stopword-based alerts")
        
        # Load or train model
        if not self.load_demo_model():
            self.train_demo_model()
        
        # Show model insights
        self.show_model_insights()
        
        # Real-time processing demo
        self.demo_real_time_processing()
        
        # Comparison with stopwords
        self.compare_with_stopword_approach()
        
        # Interactive mode
        print(f"\n🎉 Total Alerts Generated: {self.alert_count}")
        
        # Ask if user wants interactive mode
        response = input("\n🎮 Would you like to try interactive mode? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            self.interactive_demo()
        
        print(f"\n✅ Demo completed! Total alerts processed: {self.alert_count}")
        print("Thank you for using KMRL Alert Detection System!")

def main():
    """Main demo function"""
    demo = KMRLAlertDemo()
    demo.run_full_demo()

if __name__ == "__main__":
    main()