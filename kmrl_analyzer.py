#!/usr/bin/env python3
"""
KMRL Complete Alert Analysis System
Single file with all integrated processes: Classification, Translation, Tags, Database Integration

Usage:
    python kmrl_analyzer.py                    # Interactive mode
    python kmrl_analyzer.py --text "alert"     # Direct analysis
    python kmrl_analyzer.py --file input.txt   # File analysis
    python kmrl_analyzer.py --batch            # Batch processing mode
"""

import sys
import json
import argparse
import os
import re
from datetime import datetime
import uuid
from typing import Dict, List, Optional, Tuple, Any

# ================================
# KEYWORD DATA
# ================================

ALERT_KEYWORDS = {
    'severity': {
        'critical': [
            # Emergency situations
            'emergency', 'urgent', 'critical', 'immediate', 'severe', 'serious',
            'danger', 'hazard', 'risk', 'threat', 'alarm', 'warning',
            'evacuation', 'accident', 'incident', 'crash', 'collision',
            
            # Technical emergencies
            'failure', 'malfunction', 'breakdown', 'fault', 'defect',
            'damaged', 'broken', 'stopped', 'stuck', 'jammed',
            'fire', 'smoke', 'explosion', 'leak', 'rupture',
            
            # Safety critical
            'injury', 'injured', 'hurt', 'casualty', 'death', 'fatal',
            'bleeding', 'unconscious', 'trapped', 'falling',
            
            # System failures
            'power outage', 'blackout', 'system down', 'network down',
            'signal failure', 'communication failure', 'total failure'
        ],
        'high': [
            # Equipment issues
            'malfunction', 'fault', 'error', 'problem', 'issue', 'trouble',
            'breakdown', 'failure', 'defective', 'damaged', 'worn',
            'overheating', 'vibration', 'noise', 'leak', 'crack',
            
            # Service disruption
            'delayed', 'cancelled', 'suspended', 'disrupted', 'interrupted',
            'blocked', 'obstructed', 'congested', 'overcrowded',
            
            # Maintenance needs
            'repair needed', 'replace', 'service required', 'inspection needed',
            'maintenance due', 'worn out', 'expired', 'outdated',
            
            # Weather/Environmental
            'flooding', 'waterlogged', 'heavy rain', 'storm', 'lightning',
            'extreme heat', 'cold weather', 'fog', 'poor visibility'
        ],
        'medium': [
            # Routine maintenance
            'maintenance', 'service', 'inspection', 'check', 'review',
            'cleaning', 'lubrication', 'calibration', 'adjustment',
            'replacement scheduled', 'routine work', 'preventive',
            
            # Minor issues
            'minor fault', 'slight delay', 'temporary', 'intermittent',
            'occasional', 'periodic', 'fluctuation', 'variation',
            
            # Monitoring
            'monitoring', 'observation', 'tracking', 'surveillance',
            'assessment', 'evaluation', 'testing', 'verification'
        ],
        'low': [
            # Informational
            'information', 'notice', 'announcement', 'update', 'status',
            'report', 'log', 'record', 'documentation', 'note',
            
            # Routine operations
            'normal', 'routine', 'scheduled', 'planned', 'regular',
            'standard', 'typical', 'usual', 'expected', 'ongoing'
        ]
    },
    'department': {
        'operations': [
            # Train operations
            'train', 'service', 'operation', 'running', 'schedule',
            'timetable', 'route', 'journey', 'trip', 'passenger',
            'commuter', 'traveler', 'boarding', 'alighting',
            
            # Station operations
            'station', 'platform', 'concourse', 'entrance', 'exit',
            'gate', 'barrier', 'turnstile', 'ticket', 'fare',
            'crowd', 'queue', 'waiting', 'announcement', 'display',
            
            # Service management
            'delay', 'cancellation', 'disruption', 'diversion',
            'emergency stop', 'holding', 'dispatch', 'departure',
            'arrival', 'dwell time', 'headway', 'frequency'
        ],
        'maintenance': [
            # Technical maintenance
            'maintenance', 'repair', 'service', 'inspection', 'check',
            'overhaul', 'replacement', 'installation', 'upgrade',
            'modification', 'adjustment', 'calibration', 'testing',
            
            # Equipment maintenance
            'motor', 'engine', 'brake', 'wheel', 'axle', 'bearing',
            'coupling', 'door', 'window', 'seat', 'lighting',
            'air conditioning', 'ventilation', 'heating', 'cooling',
            
            # Infrastructure maintenance
            'track', 'rail', 'sleeper', 'ballast', 'bridge', 'tunnel',
            'overhead line', 'catenary', 'pantograph', 'transformer',
            'substation', 'feeder', 'cable', 'wire'
        ],
        'safety': [
            # Safety incidents
            'safety', 'security', 'accident', 'incident', 'injury',
            'casualty', 'emergency', 'evacuation', 'rescue',
            'first aid', 'medical', 'ambulance', 'fire', 'smoke',
            
            # Security issues
            'suspicious', 'unauthorized', 'trespassing', 'vandalism',
            'theft', 'robbery', 'assault', 'harassment', 'disturbance',
            'unattended bag', 'security threat', 'bomb threat',
            
            # Safety equipment
            'emergency brake', 'fire extinguisher', 'alarm', 'siren',
            'emergency lighting', 'exit sign', 'safety barrier',
            'warning sign', 'caution tape', 'protective equipment'
        ],
        'electrical': [
            # Power systems
            'power', 'electricity', 'voltage', 'current', 'transformer',
            'substation', 'feeder', 'cable', 'wire', 'conductor',
            'insulator', 'switch', 'breaker', 'fuse', 'relay',
            
            # Traction power
            'traction', 'overhead line', 'catenary', 'pantograph',
            'third rail', 'contact shoe', 'return current', 'earth',
            'bonding', 'grounding', 'isolation', 'sectioning',
            
            # Control systems
            'signaling', 'control', 'automation', 'scada', 'plc',
            'communication', 'data', 'network', 'fiber optic',
            'wireless', 'radio', 'gsm', 'wifi'
        ]
    },
    'context': {
        'railway': [
            # Rolling stock
            'train', 'coach', 'car', 'unit', 'wagon', 'locomotive',
            'emu', 'dmu', 'metro', 'rail', 'railway', 'railroad',
            
            # Infrastructure
            'track', 'platform', 'station', 'depot', 'yard',
            'workshop', 'shed', 'siding', 'junction', 'crossing',
            
            # Operations
            'service', 'line', 'route', 'schedule', 'timetable',
            'passenger', 'commuter', 'journey', 'trip'
        ],
        'location': [
            # KMRL stations
            'aluva', 'kalamassery', 'cochin university', 'pathadipalam',
            'edapally', 'changampuzha park', 'palarivattom', 'kaloor',
            'lissie', 'mg road', 'maharajas', 'ernakulam south',
            'kadavanthra', 'elamkulam', 'vyttila', 'thaikoodam',
            'petta', 'ernakulam town', 'kadavanthra', 'town hall',
            
            # Areas and landmarks
            'ernakulam', 'kochi', 'cochin', 'kerala', 'malabar',
            'airport', 'seaport', 'marine drive', 'broadway',
            'vytilla hub', 'kakkanad', 'infopark'
        ],
        'time': [
            # Time indicators
            'morning', 'afternoon', 'evening', 'night', 'midnight',
            'dawn', 'dusk', 'peak hours', 'rush hour', 'off-peak',
            'weekend', 'weekday', 'holiday', 'festival',
            
            # Temporal words
            'now', 'today', 'tomorrow', 'yesterday', 'urgent',
            'immediate', 'soon', 'later', 'ongoing', 'continuous',
            'temporary', 'permanent', 'scheduled', 'unscheduled'
        ]
    }
}

# ================================
# TRANSLATION MODULE
# ================================

class KMRLTranslator:
    """Malayalam-English translation service for KMRL alerts"""
    
    def __init__(self):
        self.translator = None
        self.cache = {}
        self._load_translator()
    
    def _load_translator(self):
        """Load Google Translate if available"""
        try:
            from googletrans import Translator
            self.translator = Translator()
            return True
        except ImportError:
            print("ℹ️  Translation service not available. Install: pip install googletrans==4.0.0-rc1")
            return False
        except Exception:
            return False
    
    def is_available(self):
        """Check if translation service is available"""
        return self.translator is not None
    
    def translate_malayalam_to_english(self, text: str) -> Dict[str, Any]:
        """Translate Malayalam text to English"""
        if not self.is_available():
            return {'translation_success': False, 'error': 'Translation service not available'}
        
        # Check cache
        if text in self.cache:
            return self.cache[text]
        
        try:
            # Detect language and translate
            detection = self.translator.detect(text)
            
            if detection.lang in ['ml', 'malayalam']:
                result = self.translator.translate(text, src='ml', dest='en')
                
                translation_data = {
                    'translation_success': True,
                    'original_text': text,
                    'translated_text': result.text,
                    'detected_language': detection.lang,
                    'confidence': detection.confidence
                }
            else:
                # Already in English or other language
                translation_data = {
                    'translation_success': True,
                    'original_text': text,
                    'translated_text': text,
                    'detected_language': detection.lang,
                    'confidence': detection.confidence,
                    'note': 'Text appears to be in English already'
                }
            
            # Cache the result
            self.cache[text] = translation_data
            return translation_data
            
        except Exception as e:
            return {
                'translation_success': False,
                'error': str(e),
                'original_text': text
            }
    
    def translate_english_to_malayalam(self, text: str) -> Dict[str, Any]:
        """Translate English text to Malayalam"""
        if not self.is_available():
            return {'translation_success': False, 'error': 'Translation service not available'}
        
        try:
            result = self.translator.translate(text, src='en', dest='ml')
            return {
                'translation_success': True,
                'original_text': text,
                'translated_text': result.text
            }
        except Exception as e:
            return {
                'translation_success': False,
                'error': str(e),
                'original_text': text
            }

# ================================
# KEYWORD CLASSIFIER
# ================================

class KeywordBasedClassifier:
    """KMRL Keyword-based alert classifier with tag generation"""
    
    def __init__(self):
        self.keywords = ALERT_KEYWORDS
        self.severity_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        print(f"📊 Keyword classifier loaded: {self._count_keywords()} total keywords")
    
    def _count_keywords(self):
        """Count total keywords"""
        total = 0
        for category in self.keywords.values():
            for subcategory in category.values():
                total += len(subcategory)
        return total
    
    def _find_keywords(self, text: str) -> Dict[str, Dict[str, List[str]]]:
        """Find matching keywords in text"""
        text_lower = text.lower()
        matches = {}
        
        for category_name, category_keywords in self.keywords.items():
            matches[category_name] = {}
            for subcategory_name, keywords in category_keywords.items():
                found_keywords = []
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        found_keywords.append(keyword)
                if found_keywords:
                    matches[category_name][subcategory_name] = found_keywords
        
        return matches
    
    def _calculate_scores(self, matched_keywords: Dict) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate severity and department scores"""
        severity_scores = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        department_scores = {}
        
        # Calculate severity scores
        if 'severity' in matched_keywords:
            for severity, keywords in matched_keywords['severity'].items():
                score = len(keywords) * self.severity_weights[severity]
                severity_scores[severity] = score
        
        # Calculate department scores
        if 'department' in matched_keywords:
            for dept, keywords in matched_keywords['department'].items():
                department_scores[dept] = len(keywords) * 2  # Base score
        
        return severity_scores, department_scores
    
    def _apply_context_boost(self, severity_scores: Dict, department_scores: Dict, 
                           matched_keywords: Dict) -> float:
        """Apply context-based score boost"""
        boost = 0
        
        # Railway context boost
        if 'context' in matched_keywords:
            if 'railway' in matched_keywords['context']:
                boost += len(matched_keywords['context']['railway']) * 0.3
            if 'location' in matched_keywords['context']:
                boost += len(matched_keywords['context']['location']) * 0.5
            if 'time' in matched_keywords['context']:
                boost += len(matched_keywords['context']['time']) * 0.2
        
        return boost
    
    def _generate_tags(self, text: str, matched_keywords: Dict, 
                      severity: str, department: str, confidence: float) -> Dict[str, List[str]]:
        """Generate search-optimized tags for document indexing and retrieval"""
        
        # Focus on search-oriented tags only
        search_keywords = set()
        
        # 1. Core classification terms (most important for search)
        search_keywords.add(severity.lower())
        search_keywords.add(department.lower())
        
        # 2. Extract all matched keywords (actual words found in text)
        for category in matched_keywords.values():
            for keywords_list in category.values():
                for keyword in keywords_list:
                    # Add original keyword
                    search_keywords.add(keyword.lower())
                    # Add normalized version for better search
                    normalized = re.sub(r'[^\w\s]', '', keyword.lower())
                    search_keywords.add(normalized)
        
        # 3. Extract specific entities and identifiers
        # Coach/Platform/Car numbers
        entity_patterns = [
            r'\b(?:coach|platform|car|unit|track|line)\s*(\d+)\b',
            r'\b(?:train|service)\s*(\w+)\b',
            r'\b(\d{1,4})\b'  # Any numbers that might be IDs
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.isdigit() and len(match) <= 4:  # Reasonable ID length
                    search_keywords.add(f"id_{match}")
                    search_keywords.add(match)
        
        # 4. KMRL-specific locations and stations
        kmrl_locations = [
            'aluva', 'kalamassery', 'edapally', 'kaloor', 'lissie', 'mg road',
            'maharajas', 'ernakulam south', 'kadavanthra', 'elamkulam', 'vyttila',
            'cochin university', 'pathadipalam', 'changampuzha park', 'palarivattom'
        ]
        
        text_lower = text.lower()
        for location in kmrl_locations:
            if location in text_lower:
                search_keywords.add(location.replace(' ', '_'))
                search_keywords.add(location)
        
        # Add general location terms
        location_terms = ['station', 'platform', 'depot', 'yard', 'track', 'line']
        for term in location_terms:
            if term in text_lower:
                search_keywords.add(term)
        
        # 5. Technical/Equipment keywords for troubleshooting search
        technical_search_terms = [
            'brake', 'door', 'engine', 'motor', 'wheel', 'signal', 'power',
            'electrical', 'mechanical', 'hydraulic', 'pneumatic', 'computer',
            'software', 'hardware', 'system', 'control', 'sensor', 'cable',
            'battery', 'transformer', 'circuit', 'switch', 'relay'
        ]
        
        for term in technical_search_terms:
            if term in text_lower:
                search_keywords.add(term)
                search_keywords.add(f"tech_{term}")
        
        # 6. Action/Status keywords for filtering documents by action type
        action_keywords = [
            'repair', 'replace', 'fix', 'maintain', 'inspect', 'clean',
            'adjust', 'calibrate', 'test', 'check', 'monitor', 'update',
            'install', 'remove', 'service', 'overhaul'
        ]
        
        for action in action_keywords:
            if action in text_lower:
                search_keywords.add(f"action_{action}")
                search_keywords.add(action)
        
        # 7. Time-sensitive keywords for chronological search
        time_indicators = [
            'immediate', 'urgent', 'emergency', 'routine', 'scheduled',
            'daily', 'weekly', 'monthly', 'annual', 'preventive',
            'corrective', 'breakdown', 'failure'
        ]
        
        for indicator in time_indicators:
            if indicator in text_lower:
                search_keywords.add(f"timing_{indicator}")
                search_keywords.add(indicator)
        
        # 8. Severity-based search tags
        severity_search_tags = {
            'critical': ['critical', 'emergency', 'urgent', 'immediate', 'crisis'],
            'high': ['high', 'important', 'priority', 'significant'],
            'medium': ['medium', 'moderate', 'standard', 'routine'],
            'low': ['low', 'minor', 'info', 'notice']
        }
        
        if severity in severity_search_tags:
            search_keywords.update(severity_search_tags[severity])
        
        # 9. Department-specific search terms
        dept_search_terms = {
            'operations': ['ops', 'service', 'passenger', 'schedule', 'delay'],
            'maintenance': ['maint', 'repair', 'technical', 'equipment', 'spare'],
            'safety': ['safety', 'security', 'incident', 'accident', 'hazard'],
            'electrical': ['electrical', 'power', 'voltage', 'current', 'circuit']
        }
        
        if department in dept_search_terms:
            search_keywords.update(dept_search_terms[department])
        
        # 10. Add composite search terms for better findability
        search_keywords.add(f"{severity}_{department}")
        search_keywords.add(f"conf_{int(confidence * 100)}")
        
        # 11. Extract and normalize all significant words from text
        # Remove common stop words but keep technical terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Extract all words, filter stop words, keep meaningful terms
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        for word in words:
            if word not in stop_words and len(word) >= 3:
                search_keywords.add(word)
        
        # Clean and optimize for search
        final_search_tags = []
        for tag in search_keywords:
            if tag and len(str(tag).strip()) >= 2:  # Minimum tag length
                # Clean the tag
                clean_tag = re.sub(r'[^\w\-_]', '', str(tag).lower().strip())
                if clean_tag and clean_tag not in final_search_tags:
                    final_search_tags.append(clean_tag)
        
        # Sort by relevance - put most important search terms first
        priority_order = []
        regular_tags = []
        
        for tag in final_search_tags:
            if any(tag.startswith(prefix) for prefix in [severity, department, 'id_', 'action_', 'tech_']):
                priority_order.append(tag)
            else:
                regular_tags.append(tag)
        
        # Return optimized search tags
        return {
            'search_tags': (priority_order + regular_tags)[:20],  # Limit to top 20 for performance
            'matched_keywords': list(set([kw for cat in matched_keywords.values() for kwlist in cat.values() for kw in kwlist])),
            'entity_tags': [tag for tag in final_search_tags if tag.startswith('id_') or tag.isdigit()],
            'location_tags': [tag for tag in final_search_tags if any(loc in tag for loc in kmrl_locations + location_terms)],
            'technical_tags': [tag for tag in final_search_tags if tag.startswith('tech_')],
            'action_tags': [tag for tag in final_search_tags if tag.startswith('action_')],
            'priority_tags': priority_order[:5]  # Most important for quick search
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict alert severity and department with comprehensive analysis"""
        # Find matching keywords
        matched_keywords = self._find_keywords(text)
        
        # Calculate base scores
        severity_scores, department_scores = self._calculate_scores(matched_keywords)
        
        # Apply context boost
        context_boost = self._apply_context_boost(severity_scores, department_scores, matched_keywords)
        
        # Boost scores with context
        for severity in severity_scores:
            severity_scores[severity] += context_boost * 0.3
        
        for dept in department_scores:
            department_scores[dept] += context_boost * 0.2
        
        # Determine predictions
        predicted_severity = max(severity_scores, key=severity_scores.get) if any(severity_scores.values()) else 'low'
        predicted_department = max(department_scores, key=department_scores.get) if any(department_scores.values()) else 'operations'
        
        # Calculate confidence
        max_severity_score = severity_scores[predicted_severity]
        total_severity_score = sum(severity_scores.values())
        severity_confidence = max_severity_score / max(total_severity_score, 1)
        
        max_dept_score = department_scores[predicted_department] if department_scores else 0
        total_dept_score = sum(department_scores.values()) if department_scores else 1
        department_confidence = max_dept_score / max(total_dept_score, 1)
        
        overall_confidence = (severity_confidence + department_confidence) / 2
        
        # Generate explanation
        explanation_parts = []
        if 'severity' in matched_keywords:
            for sev, keywords in matched_keywords['severity'].items():
                explanation_parts.append(f"'{sev.title()}' severity keywords found: {', '.join(keywords)}")
        
        if 'department' in matched_keywords:
            for dept, keywords in matched_keywords['department'].items():
                explanation_parts.append(f"'{dept.title()}' department keywords found: {', '.join(keywords)}")
        
        if 'context' in matched_keywords:
            for ctx, keywords in matched_keywords['context'].items():
                explanation_parts.append(f"Railway context ({ctx}) keywords: {', '.join(keywords)}")
        
        explanation_parts.append(f"Predicted: {predicted_severity.title()} severity, {predicted_department.title()} department")
        explanation_parts.append(f"Confidence: {overall_confidence*100:.1f}%")
        
        explanation = " | ".join(explanation_parts)
        
        # Generate comprehensive tags
        tags = self._generate_tags(text, matched_keywords, predicted_severity, predicted_department, overall_confidence)
        
        return {
            'severity': predicted_severity,
            'department': predicted_department,
            'confidence': overall_confidence,
            'severity_confidence': severity_confidence,
            'department_confidence': department_confidence,
            'context_boost': context_boost,
            'matched_keywords': matched_keywords,
            'severity_scores': severity_scores,
            'department_scores': department_scores,
            'explanation': explanation,
            'tags': tags
        }

# ================================
# DATABASE INTEGRATION
# ================================

class KMRLDatabaseProcessor:
    """Database integration for KMRL alerts"""
    
    def __init__(self):
        self.classifier = KeywordBasedClassifier()
        self.translator = KMRLTranslator()
    
    def process_single_document(self, text: str, metadata: Dict = None) -> Dict[str, Any]:
        """Process a single document for database insertion"""
        # Generate unique ID
        document_id = str(uuid.uuid4())
        
        # Classify the text
        classification = self.classifier.predict(text)
        
        # Prepare database record
        record = {
            # Unique identifiers
            "id": document_id,
            "timestamp": datetime.now().isoformat(),
            "processing_date": datetime.now().date().isoformat(),
            "processing_time": datetime.now().time().isoformat(),
            
            # Document content
            "document_text": text,
            "document_length": len(text),
            "source": metadata.get('source', 'manual_input') if metadata else 'manual_input',
            
            # Classification results
            "severity": classification.get('severity', 'unknown'),
            "department": classification.get('department', 'unknown'),
            "confidence": round(float(classification.get('confidence', 0)), 2),
            "severity_confidence": round(float(classification.get('severity_confidence', 0)), 2),
            "department_confidence": round(float(classification.get('department_confidence', 0)), 2),
            
            # Analysis details
            "matched_keywords": classification.get('matched_keywords', {}),
            "keyword_count": sum(len(keywords) for keywords in classification.get('matched_keywords', {}).values()),
            "context_boost": round(float(classification.get('context_boost', 0)), 2),
            "explanation": classification.get('explanation', ''),
            "tags": classification.get('tags', {}),
            
            # Processing metadata
            "model_type": "keyword_based",
            "model_version": "1.0",
            "processing_engine": "kmrl_analyzer",
            "status": "processed",
            "processing_success": True,
            "metadata": metadata or {}
        }
        
        return {
            "success": True,
            "document_id": document_id,
            "record": record,
            "classification": classification
        }

# ================================
# MAIN ANALYZER CLASS
# ================================

class KMRLAnalyzer:
    """Complete KMRL Alert Analysis System"""
    
    def __init__(self):
        self.classifier = KeywordBasedClassifier()
        self.translator = KMRLTranslator()
        self.db_processor = KMRLDatabaseProcessor()
        print("✅ KMRL Analyzer initialized successfully!")
    
    def analyze_comprehensive(self, text: str, include_translation: bool = False, 
                            translate_input: bool = False, translate_output: bool = False, minimal: bool = False) -> Dict[str, Any]:
        """Comprehensive analysis with all features"""
        
        original_text = text
        translation_details = {}
        
        # Handle input translation
        if translate_input and self.translator.is_available():
            trans_result = self.translator.translate_malayalam_to_english(text)
            if trans_result['translation_success']:
                text = trans_result['translated_text']
                if not minimal:
                    translation_details['input_translation'] = trans_result
        
        # Perform classification
        classification = self.classifier.predict(text)
        
        if minimal:
            # Minimal output with search-focused tags
            search_tags = classification['tags'].get('priority_tags', [])
            if len(search_tags) < 5:
                # Fill up with most relevant search tags
                additional_tags = classification['tags'].get('search_tags', [])
                search_tags.extend(additional_tags[:5-len(search_tags)])
            
            result = {
                "alert_id": f"KMRL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "severity": classification['severity'].upper(),
                "department": classification['department'].upper(),
                "confidence": round(classification['confidence'] * 100, 1),
                "priority": self._get_priority_level(classification['severity'], classification['confidence']),
                "search_tags": search_tags[:5],  # Top 5 search-optimized tags
                "immediate_action": classification['severity'] in ['critical', 'high'],
                "response_time": self._get_response_time(classification['severity']).replace('_', ' '),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add translation if available
            if translate_output and self.translator.is_available():
                trans_result = self.translator.translate_english_to_malayalam(f"{classification['severity']} alert in {classification['department']} department")
                if trans_result['translation_success']:
                    result['malayalam_alert'] = trans_result['translated_text']
            
            return result
        
        # Full comprehensive result for non-minimal mode
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "processing_info": {
                "input_length": len(text),
                "translation_available": self.translator.is_available(),
                "analysis_type": "comprehensive_keyword_detection"
            }
        }
        
        # Build comprehensive result
        result.update({
            "input_text": original_text,
            "processed_text": text,
            "classification": {
                "severity": classification['severity'],
                "department": classification['department'],
                "confidence": round(classification['confidence'], 3),
                "severity_confidence": round(classification['severity_confidence'], 3),
                "department_confidence": round(classification['department_confidence'], 3)
            },
            "alert_details": {
                "alert_id": f"KMRL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "priority_level": self._get_priority_level(classification['severity'], classification['confidence']),
                "urgency": classification['severity'].upper(),
                "routing_department": classification['department'].upper(),
                "confidence_score": round(classification['confidence'] * 100, 1),
                "requires_immediate_attention": classification['severity'] in ['critical', 'high'],
                "escalation_needed": classification['severity'] == 'critical' and classification['confidence'] > 0.6,
                "estimated_response_time": self._get_response_time(classification['severity']),
                "context_boost_applied": round(classification['context_boost'], 2)
            },
            "keyword_analysis": {
                "total_keywords_matched": sum(len(kw) for cat in classification['matched_keywords'].values() for kw in cat.values()),
                "matched_keywords": classification['matched_keywords'],
                "severity_scores": classification['severity_scores'],
                "department_scores": classification['department_scores'],
                "explanation": classification['explanation']
            },
            "tags": classification['tags'],
            "search_optimization": {
                "total_tags_generated": sum(len(tags) for tags in classification['tags'].values()),
                "searchable_terms": classification['tags'].get('search_tags', []),
                "entity_references": classification['tags'].get('entity_tags', []),
                "location_keywords": classification['tags'].get('location_tags', [])
            }
        })
        
        # Handle output translation
        if translate_output and self.translator.is_available():
            trans_result = self.translator.translate_english_to_malayalam(classification['explanation'])
            if trans_result['translation_success']:
                translation_details['output_translation'] = trans_result
        
        # Add translation details
        if translation_details:
            result['translation_details'] = translation_details
        
        # Generate database record
        db_result = self.db_processor.process_single_document(original_text)
        result['database_record'] = db_result['record']
        
        return result
    
    def _get_priority_level(self, severity: str, confidence: float) -> str:
        """Get priority level based on severity and confidence"""
        if severity == 'critical' and confidence > 0.7:
            return 'P1_CRITICAL'
        elif severity == 'critical' or (severity == 'high' and confidence > 0.8):
            return 'P2_HIGH'
        elif severity == 'high' or (severity == 'medium' and confidence > 0.7):
            return 'P3_MEDIUM'
        else:
            return 'P4_LOW'
    
    def _get_response_time(self, severity: str) -> str:
        """Get estimated response time"""
        times = {'critical': '15_minutes', 'high': '1_hour', 'medium': '4_hours', 'low': '24_hours'}
        return times.get(severity, '24_hours')
    
    def interactive_mode(self):
        """Interactive terminal mode"""
        print("🚀 KMRL Interactive Alert Analysis System")
        print("Type 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                # Get input
                text = input("📝 Enter alert text: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not text:
                    print("❌ Please enter some text\n")
                    continue
                
                # Ask for translation
                try:
                    translate_input = input("🌐 Include translation? (y/n): ").strip().lower()
                    translate = translate_input in ['y', 'yes']
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Goodbye!")
                    break
                
                print("\n🔍 Analyzing...")
                
                # Analyze
                try:
                    result = self.analyze_comprehensive(
                        text, 
                        include_translation=translate,
                        translate_input=translate,
                        translate_output=translate,
                        minimal=True  # Always use minimal output in interactive mode
                    )
                    
                    # Display results
                    print(f"\n📊 ANALYSIS RESULTS")
                    print(f"{'='*50}")
                    print(f"🚨 Severity: {result['severity']}")
                    print(f"🏢 Department: {result['department']}")
                    print(f"📈 Confidence: {result['confidence']}%")
                    print(f"⚡ Priority: {result['priority']}")
                    print(f"⏱️  Response Time: {result['response_time']}")
                    
                    if result['immediate_action']:
                        print("⚠️  🚨 IMMEDIATE ACTION REQUIRED!")
                    
                    # Show tags
                    if result.get('search_tags'):
                        print(f"\n🔍 Search Keywords: {', '.join(result['search_tags'])}")
                    
                    # Show translation if available
                    if 'malayalam_alert' in result:
                        print(f"\n🌐 Malayalam: {result['malayalam_alert']}")
                    
                    print(f"\n💾 Alert ID: {result['alert_id']}")
                    print(f"{'='*50}\n")
                    
                except Exception as e:
                    print(f"❌ Analysis Error: {e}\n")
                
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")

# ================================
# COMMAND LINE INTERFACE
# ================================

def main():
    parser = argparse.ArgumentParser(
        description='KMRL Complete Alert Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python kmrl_analyzer.py                                    # Interactive mode
    python kmrl_analyzer.py --text "Emergency at station"     # Direct analysis
    python kmrl_analyzer.py --file alerts.txt                 # File analysis
    python kmrl_analyzer.py --text "alert" --translate        # With translation
    python kmrl_analyzer.py --text "alert" --json             # JSON output
    python kmrl_analyzer.py --batch                           # Batch mode
        '''
    )
    
    parser.add_argument('--text', help='Text to analyze directly')
    parser.add_argument('--file', help='File containing text to analyze')
    parser.add_argument('--translate', action='store_true', help='Include translation features')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    parser.add_argument('--compact', action='store_true', help='Compact JSON output')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--minimal', action='store_true', help='Minimal output with only essential information')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = KMRLAnalyzer()
    
    # Check if input is piped (non-interactive)
    import sys
    if not sys.stdin.isatty() and not any([args.text, args.file, args.batch]):
        # Handle piped input
        try:
            input_data = sys.stdin.read().strip()
            if input_data and input_data.lower() not in ['quit', 'exit']:
                result = analyzer.analyze_comprehensive(
                    input_data,
                    include_translation=args.translate,
                    translate_input=args.translate,
                    translate_output=args.translate,
                    minimal=True
                )
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, indent=None if args.compact else 2))
                else:
                    print(f"🚨 Severity: {result['severity']}")
                    print(f"🏢 Department: {result['department']}")
                    print(f"📈 Confidence: {result['confidence']}%")
        except Exception as e:
            print(f"❌ Error processing piped input: {e}")
        return
    
    if args.batch:
        # Batch processing mode
        print("📦 Batch Processing Mode")
        print("Enter texts one by one. Type 'DONE' to finish.\n")
        
        results = []
        counter = 1
        
        while True:
            text = input(f"Text {counter}: ").strip()
            if text.upper() == 'DONE':
                break
            
            if text:
                result = analyzer.analyze_comprehensive(text, include_translation=args.translate, minimal=True)
                results.append(result)
                print(f"✅ Processed #{counter}: {result['severity']}")
                counter += 1
        
        # Output batch results
        batch_output = {
            "batch_id": f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "total_processed": len(results),
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        if args.json:
            print(json.dumps(batch_output, indent=None if args.compact else 2, ensure_ascii=False))
        else:
            print(f"\n📊 Batch Summary: {len(results)} alerts processed")
        
    elif args.text:
        # Direct text analysis
        result = analyzer.analyze_comprehensive(
            args.text, 
            include_translation=args.translate,
            translate_input=args.translate,
            translate_output=args.translate,
            minimal=args.minimal or args.json  # Use minimal for JSON output by default
        )
        
        if args.json:
            print(json.dumps(result, indent=None if args.compact else 2, ensure_ascii=False))
        else:
            # Human readable output
            if hasattr(result, 'get') and 'severity' in result:
                print(f"🚨 Severity: {result['severity']}")
                print(f"🏢 Department: {result['department']}")
                print(f"📈 Confidence: {result['confidence']}%")
                print(f"⚡ Priority: {result['priority']}")
                
                if result['immediate_action']:
                    print("⚠️  🚨 IMMEDIATE ACTION REQUIRED!")
            else:
                # Fallback for full result
                print(f"🚨 Severity: {result['classification']['severity'].upper()}")
                print(f"🏢 Department: {result['classification']['department'].upper()}")
                print(f"📈 Confidence: {result['alert_details']['confidence_score']:.1f}%")
                print(f"⚡ Priority: {result['alert_details']['priority_level']}")
                
                if result['alert_details']['requires_immediate_attention']:
                    print("⚠️  🚨 IMMEDIATE ATTENTION REQUIRED!")
    
    elif args.file:
        # File analysis
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            result = analyzer.analyze_comprehensive(
                text, 
                include_translation=args.translate,
                minimal=args.minimal or args.json
            )
            
            if args.json:
                print(json.dumps(result, indent=None if args.compact else 2, ensure_ascii=False))
            else:
                print(f"📁 File: {args.file}")
                if hasattr(result, 'get') and 'severity' in result:
                    print(f"🚨 Severity: {result['severity']}")
                    print(f"🏢 Department: {result['department']}")
                    print(f"📈 Confidence: {result['confidence']}%")
                else:
                    print(f"🚨 Severity: {result['classification']['severity'].upper()}")
                    print(f"🏢 Department: {result['classification']['department'].upper()}")
                    print(f"📈 Confidence: {result['alert_details']['confidence_score']:.1f}%")
                
        except FileNotFoundError:
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
    
    else:
        # Interactive mode (default)
        analyzer.interactive_mode()

if __name__ == "__main__":
    main()