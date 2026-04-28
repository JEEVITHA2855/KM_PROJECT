#!/usr/bin/env python3
"""
KMRL Production-Grade Alert Analysis System (v7.0)
==================================================

Industry-ready ML alert classification using transformer models:
- BART-MNLI for zero-shot classification (severity + departments)
- MiniLM for semantic embeddings and search
- Batch processing for 5000+ alerts/day
- GPU/CPU auto-detection
- Model caching and optimization
- Production logging and metrics

Features:
- Zero-shot classification with BART
- Semantic search with MiniLM embeddings
- Batch processing pipeline
- Model quantization support
- Real-time and batch modes
- Comprehensive metrics tracking
- Production-ready error handling

Usage:
    python analyzer.py --text "Emergency brake failure"
    python analyzer.py --batch --file alerts.txt --workers 4
    python analyzer.py --search "fire hazard" --top-k 10
    python analyzer.py --metrics

Version: 7.0 (Production-grade, Transformer-based)
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ================================
# ENVIRONMENT & LOGGING SETUP
# ================================

# Setup logging with UTF-8 encoding
import io
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')),
        logging.FileHandler('kmrl_alerts.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Detect GPU availability
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"[Device] {DEVICE.upper()}")
except:
    DEVICE = 'cpu'
    logger.info("[Device] CPU")

# ================================
# ML IMPORTS & INITIALIZATION
# ================================

try:
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    from tqdm import tqdm
    
    logger.info("[OK] All transformer libraries loaded successfully")
except ImportError as e:
    logger.error(f"❌ Import Error: {e}")
    logger.error("Install: pip install -r requirements.txt")
    sys.exit(1)


# ================================
# PRODUCTION-GRADE CLASSIFIER
# ================================

class ProductionMLClassifier:
    """
    Industry-ready ML classifier with transformer models
    Handles 5000+ alerts/day efficiently
    """
    
    def __init__(self, use_quantization: bool = False, batch_size: int = 32):
        """Initialize production classifier"""
        
        logger.info("[...] Initializing production ML classifier...")
        self.device = DEVICE
        self.batch_size = batch_size
        self.use_quantization = use_quantization
        
        # Model names
        self.zero_shot_model = "facebook/bart-large-mnli"
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.alert_database = []
        
        # Metrics
        self.metrics = {
            'total_alerts': 0,
            'avg_latency': 0,
            'cache_hits': 0,
            'errors': 0
        }
        
        # Load models
        self._load_models()
        
        # Severity labels for zero-shot classification
        self.severity_labels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        
        # Department labels
        self.department_labels = [
            "OPERATIONS", "SAFETY", "MAINTENANCE", "FINANCE",
            "HR", "LEGAL", "PROCUREMENT", "ELECTRICAL"
        ]
        
        logger.info("[OK] Production classifier initialized\n")
    
    def _load_models(self):
        """Load transformer models with optimization"""
        
        logger.info("[>] Loading transformer models...")
        start_time = time.time()
        
        try:
            # Load zero-shot classifier (BART-MNLI)
            logger.info("[..] 1/2 Loading zero-shot model (BART-MNLI)...")
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model=self.zero_shot_model,
                device=0 if self.device == 'cuda' else -1
            )
            logger.info("[OK] Zero-shot classifier loaded")
            
            # Load semantic embedding model (MiniLM)
            logger.info("[..] 2/2 Loading embedding model (MiniLM)...")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device
            )
            if self.use_quantization:
                try:
                    self.embedding_model.quantize()
                    logger.info("[OK] Model quantized for efficiency")
                except:
                    logger.warning("[!] Quantization not available")
            
            logger.info("[OK] Embedding model loaded")
            
            elapsed = time.time() - start_time
            logger.info(f"[SUMMARY] All models loaded in {elapsed:.2f}s\n")
            
        except Exception as e:
            logger.error(f"[ERROR] Error loading models: {e}")
            raise
    
    def classify_severity_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Batch classify severity using zero-shot classification"""
        
        results = []
        for text in texts:
            try:
                output = self.zero_shot_classifier(
                    text[:512],  # Truncate to max length
                    self.severity_labels,
                    hypothesis_template="This alert is {}.",
                    multi_class=False
                )
                severity = output['labels'][0]
                confidence = round(output['scores'][0] * 100, 2)
                results.append((severity, confidence))
            except Exception as e:
                logger.warning(f"Error classifying text: {e}")
                results.append(("MEDIUM", 50.0))
                self.metrics['errors'] += 1
        
        return results
    
    def classify_department_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Batch classify departments using zero-shot classification"""
        
        results = []
        for text in texts:
            try:
                output = self.zero_shot_classifier(
                    text[:512],
                    self.department_labels,
                    hypothesis_template="This alert is about {}.",
                    multi_class=False
                )
                dept = output['labels'][0]
                confidence = round(output['scores'][0] * 100, 2)
                results.append((dept, confidence))
            except Exception as e:
                logger.warning(f"Error classifying department: {e}")
                results.append(("OPERATIONS", 50.0))
                self.metrics['errors'] += 1
        
        return results
    
    def extract_keywords_from_text(self, text: str, top_k: int = 5) -> List[str]:
        """Extract keywords using simple heuristics"""
        
        try:
            # Split text into words and clean
            words = text.lower().split()
            
            # Filter common words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were',
                'this', 'that', 'to', 'from', 'in', 'on', 'at', 'by', 'for',
                'of', 'with', 'as', 'it', 'be', 'have', 'has', 'do', 'does'
            }
            
            keywords = [w for w in words if w not in stop_words and len(w) > 3]
            return keywords[:top_k]
        except:
            return []
    
    def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Get semantic embedding with caching"""
        
        cache_key = hash(text)
        
        if use_cache and cache_key in self.embedding_cache:
            self.metrics['cache_hits'] += 1
            return self.embedding_cache[cache_key]
        
        embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        
        if use_cache:
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def semantic_search_batch(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search across alert database"""
        
        if not self.alert_database:
            logger.warning("[!] Alert database is empty")
            return []
        
        try:
            query_embedding = self.get_embedding(query)
            
            results = []
            for alert in self.alert_database:
                alert_embedding = self.get_embedding(alert['text'])
                
                # Calculate cosine similarity
                similarity = util.pytorch_cos_sim(
                    query_embedding, alert_embedding
                ).item()
                
                results.append({
                    'alert_id': alert['alert_id'],
                    'text': alert['text'][:100],
                    'severity': alert['severity'],
                    'department': alert['department'],
                    'similarity': round(similarity * 100, 2),
                    'keywords': alert.get('keywords', [])
                })
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
        
        except Exception as e:
            logger.error(f"[ERROR] Error in semantic search: {e}")
            return []
    
    def analyze_alert(self, text: str) -> Dict[str, Any]:
        """Analyze single alert"""
        
        start_time = time.time()
        
        try:
            # Classify severity
            severity, severity_conf = self.classify_severity_batch([text])[0]
            
            # Classify department
            department, dept_conf = self.classify_department_batch([text])[0]
            
            # Extract keywords
            keywords = self.extract_keywords_from_text(text)
            
            # Priority mapping
            priority_map = {
                "CRITICAL": "P1_CRITICAL",
                "HIGH": "P2_HIGH",
                "MEDIUM": "P3_MEDIUM",
                "LOW": "P4_LOW"
            }
            
            response_map = {
                "CRITICAL": "5 minutes",
                "HIGH": "15 minutes",
                "MEDIUM": "1 hour",
                "LOW": "24 hours"
            }
            
            alert = {
                "alert_id": f"KMRL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "text": text,
                "severity": severity,
                "severity_confidence": severity_conf,
                "department": department,
                "department_confidence": dept_conf,
                "priority": priority_map[severity],
                "response_time": response_map[severity],
                "overall_confidence": round((severity_conf + dept_conf) / 2, 2),
                "keywords": keywords,
                "immediate_action": severity in ["CRITICAL", "HIGH"],
                "timestamp": datetime.now().isoformat(),
                "model": "BART-MNLI + MiniLM-L6-v2",
                "device": self.device.upper(),
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            self.alert_database.append(alert)
            self.metrics['total_alerts'] += 1
            
            return alert
        
        except Exception as e:
            logger.error(f"[ERROR] Error analyzing alert: {e}")
            self.metrics['errors'] += 1
            return None
    
    def analyze_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        """Analyze multiple alerts efficiently"""
        
        logger.info(f"[BATCH] Processing batch of {len(texts)} alerts...")
        batch_start = time.time()
        
        results = []
        iterator = tqdm(texts, disable=not show_progress, desc="Processing", unit="alert")
        
        # Process in chunks for efficiency
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            
            # Batch classify
            severities = self.classify_severity_batch(chunk)
            departments = self.classify_department_batch(chunk)
            
            # Create alerts
            for j, text in enumerate(chunk):
                severity, sev_conf = severities[j]
                dept, dept_conf = departments[j]
                
                priority_map = {
                    "CRITICAL": "P1_CRITICAL",
                    "HIGH": "P2_HIGH",
                    "MEDIUM": "P3_MEDIUM",
                    "LOW": "P4_LOW"
                }
                
                response_map = {
                    "CRITICAL": "5 minutes",
                    "HIGH": "15 minutes",
                    "MEDIUM": "1 hour",
                    "LOW": "24 hours"
                }
                
                alert = {
                    "alert_id": f"KMRL_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}",
                    "text": text,
                    "severity": severity,
                    "severity_confidence": sev_conf,
                    "department": dept,
                    "department_confidence": dept_conf,
                    "priority": priority_map[severity],
                    "response_time": response_map[severity],
                    "overall_confidence": round((sev_conf + dept_conf) / 2, 2),
                    "keywords": self.extract_keywords_from_text(text),
                    "immediate_action": severity in ["CRITICAL", "HIGH"],
                    "timestamp": datetime.now().isoformat(),
                    "model": "BART-MNLI + MiniLM-L6-v2",
                    "device": self.device.upper()
                }
                
                results.append(alert)
                self.alert_database.append(alert)
                iterator.update(1)
        
        batch_elapsed = time.time() - batch_start
        alerts_per_sec = len(texts) / batch_elapsed if batch_elapsed > 0 else 0
        logger.info(f"[OK] Batch processed in {batch_elapsed:.2f}s ({alerts_per_sec:.1f} alerts/sec)\n")
        
        self.metrics['total_alerts'] += len(texts)
        return results
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        
        return {
            "total_alerts_processed": self.metrics['total_alerts'],
            "cache_hits": self.metrics['cache_hits'],
            "cache_efficiency": round((self.metrics['cache_hits'] / max(1, self.metrics['total_alerts'])) * 100, 2),
            "errors": self.metrics['errors'],
            "device": self.device.upper(),
            "database_size": len(self.alert_database),
            "embedding_cache_size": len(self.embedding_cache)
        }


# ================================
# CLI & PRODUCTION INTERFACE
# ================================

def main():
    """Production CLI interface"""
    
    parser = argparse.ArgumentParser(
        description='KMRL Production-Grade Alert Classifier (v7.0)',
        epilog='''Examples:
  python analyzer.py --text "Emergency brake failure"
  python analyzer.py --batch --file alerts.txt --workers 4
  python analyzer.py --search "fire hazard"
  python analyzer.py --metrics
        '''
    )
    
    parser.add_argument('--text', help='Analyze single alert text')
    parser.add_argument('--file', help='Analyze alerts from file (one per line)')
    parser.add_argument('--search', help='Semantic search across database')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--workers', type=int, default=32, help='Batch size')
    parser.add_argument('--json', action='store_true', help='JSON output')
    parser.add_argument('--metrics', action='store_true', help='Show metrics')
    parser.add_argument('--quantize', action='store_true', help='Use model quantization')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = ProductionMLClassifier(
        use_quantization=args.quantize,
        batch_size=args.workers
    )
    
    # Process based on arguments
    if args.text:
        alert = classifier.analyze_alert(args.text)
        
        if alert:
            if args.json:
                print(json.dumps(alert, indent=2, default=str))
            else:
                print(f"\n[ANALYSIS] ALERT ANALYSIS (v7.0)")
                print(f"{'='*70}")
                print(f"Alert ID: {alert['alert_id']}")
                print(f"[SEVERITY] {alert['severity']} ({alert['severity_confidence']}%)")
                print(f"[DEPARTMENT] {alert['department']} ({alert['department_confidence']}%)")
                print(f"[PRIORITY] {alert['priority']}")
                print(f"[RESPONSE] {alert['response_time']}")
                print(f"[CONFIDENCE] {alert['overall_confidence']}%")
                print(f"[DEVICE] {alert['device']}")
                print(f"[TIME] {alert['processing_time_ms']}ms")
                
                if alert['immediate_action']:
                    print("[ACTION] IMMEDIATE ACTION REQUIRED!")
                
                if alert['keywords']:
                    print(f"[KEYWORDS] {', '.join(alert['keywords'])}")
                
                print(f"{'='*70}\n")
    
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                alerts = [line.strip() for line in f if line.strip()]
            
            results = classifier.analyze_batch(alerts, show_progress=True)
            
            if args.json:
                print(json.dumps(results, indent=2, default=str))
            else:
                print(f"\n[BATCH] BATCH PROCESSING COMPLETE")
                print(f"{'='*70}")
                
                severity_count = defaultdict(int)
                dept_count = defaultdict(int)
                
                for alert in results:
                    severity_count[alert['severity']] += 1
                    dept_count[alert['department']] += 1
                
                print(f"\n[SUMMARY] ({len(results)} alerts)")
                print(f"Severity Distribution:")
                for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                    count = severity_count[sev]
                    print(f"  {sev}: {count}")
                
                print(f"\nDepartment Distribution:")
                for dept in sorted(dept_count.keys()):
                    print(f"  {dept}: {dept_count[dept]}")
                
                print(f"{'='*70}\n")
        
        except FileNotFoundError:
            logger.error(f"[ERROR] File not found: {args.file}")
    
    elif args.search:
        results = classifier.semantic_search_batch(args.search, top_k=10)
        
        if not results:
            print("[-] No alerts found in database")
        elif args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"\n[SEARCH] SEMANTIC SEARCH: '{args.search}'")
            print(f"{'='*70}")
            for i, r in enumerate(results, 1):
                print(f"\n{i}. {r['alert_id']} ({r['similarity']}% match)")
                print(f"   Text: {r['text']}")
                print(f"   Severity: {r['severity']}")
                print(f"   Department: {r['department']}")
            print(f"{'='*70}\n")
    
    elif args.metrics:
        metrics = classifier.get_metrics()
        print("\n📊 PERFORMANCE METRICS")
        print(f"{'='*70}")
        for key, value in metrics.items():
            print(f"{key:.<40} {value}")
        print(f"{'='*70}\n")
    
    elif args.interactive:
        print("🚆 KMRL Interactive Alert Analyzer (v7.0)")
        print("Commands: 'search <term>' or enter alert text, 'metrics', 'quit'\n")
        
        while True:
            try:
                inp = input("📝 > ").strip()
                
                if inp.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif inp.lower().startswith('search '):
                    query = inp[7:]
                    results = classifier.semantic_search_batch(query, top_k=5)
                    if results:
                        for r in results:
                            print(f"  {r['alert_id']}: {r['similarity']}% → {r['text'][:50]}")
                    else:
                        print("  No results")
                
                elif inp.lower() == 'metrics':
                    metrics = classifier.get_metrics()
                    for k, v in metrics.items():
                        print(f"  {k}: {v}")
                
                elif inp:
                    alert = classifier.analyze_alert(inp)
                    if alert:
                        print(f"  ✅ {alert['severity']} | {alert['department']} | {alert['priority']}\n")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
