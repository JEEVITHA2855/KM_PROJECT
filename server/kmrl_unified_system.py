#!/usr/bin/env python3
"""
KMRL Unified ML System
=====================

Complete enterprise ML system combining:
1. Alert Classification (severity, type, department)
2. Semantic Search (find similar documents using embeddings)

Features:
---------
* Enterprise Alert Classification: Multi-label ML classification
* Semantic Document Search: Find semantically similar documents
* Production Ready: High performance with caching and metrics
* Unified Interface: Single system for both functionalities
* JSON API: REST endpoints for system integration

Usage:
------
    # Interactive mode (recommended)
    python kmrl_unified_system.py
    
    # Then use commands:
    # classify <text>     - Classify alert text
    # search <query>     - Search for similar documents  
    # index <file>       - Index documents from file
    # metrics           - Show system metrics
    # exit              - Exit program

Author: KMRL Analytics Team
Version: 6.0 (Unified System)
License: MIT
"""

import os
import sys
import json
import time
import hashlib
import uuid
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

# Flask imports for API functionality
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Core Dependencies
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] KMRL.Unified: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# =============================================================================
# ALERT CLASSIFICATION DATA STRUCTURES
# =============================================================================

@dataclass
class AlertClassification:
    """Alert classification result"""
    severity: str
    alert_type: str
    department: str
    confidence_scores: Dict[str, float]
    risk_level: str
    processing_time_ms: float

@dataclass
class AlertResponse:
    """Simplified alert classification response"""
    status: str
    severity: str
    alert_type: str
    department: str
    keywords: List[str]
    confidence: float
    processing_time: float
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class EmbeddingResponse:
    """Text embedding response for API service"""
    status: str
    embedding: List[float]
    model: str
    dimension: int
    processing_time: float
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def validate_embedding(self) -> bool:
        """Validate embedding has correct dimensions and values"""
        return (
            len(self.embedding) == int(self.dimension) and
            all(isinstance(x, (int, float)) for x in self.embedding) and
            any(abs(x) > 1e-6 for x in self.embedding)  # Not all zeros
        )

# =============================================================================
# SEMANTIC SEARCH DATA STRUCTURES  
# =============================================================================

@dataclass
class SearchResult:
    """Individual search result with similarity score"""
    document_id: str
    content: str
    similarity_score: float
    rank: int
    snippet: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class SearchResponse:
    """Complete search response"""
    query: str
    timestamp: str
    total_results: int
    search_time_ms: float
    results: List[SearchResult]
    expanded_terms: List[str]
    search_metadata: Dict[str, Any]
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

# =============================================================================
# UNIFIED ML SYSTEM
# =============================================================================

class KMRLUnifiedMLSystem:
    """
    Unified ML system combining alert classification and semantic search
    """
    
    def __init__(self):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load sentence embedding model (true model-based embeddings)
        self.embedding_model_name = os.getenv(
            "KMRL_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        try:
            self.embedder = SentenceTransformer(
                self.embedding_model_name,
                device=("cuda" if torch.cuda.is_available() else "cpu"),
            )
        except Exception as e:
            raise RuntimeError(f"Error loading embedding model '{self.embedding_model_name}': {e}")

        self.embedding_dim = int(self.embedder.get_sentence_embedding_dimension())
        
        # Performance metrics
        self.metrics = {
            "alert_classifications": 0,
            "semantic_searches": 0,
            "avg_alert_time": 0,
            "avg_search_time": 0,
            "total_indexed_docs": 0,
            "cache_hits": 0
        }
        
        # Initialize alert classification system
        self._init_alert_classifier()
        
        # Initialize semantic search system
        self._init_semantic_search()
        
        print("✓ KMRL Unified ML System initialized")
        print("=" * 60)
    
    def _init_alert_classifier(self):
        """Initialize model-based classification components (no keyword rules)."""

        # Candidate label descriptions (used for embedding-similarity classification)
        # Note: the label *names* are fixed, but prediction is ML-based.
        self._severity_labels = {
            "CRITICAL": "Immediate danger to life/safety or major incident requiring urgent response.",
            "HIGH": "Major operational issue requiring prompt attention; service impact likely.",
            "MEDIUM": "Moderate issue; needs attention but not immediately life-threatening.",
            "LOW": "Minor / informational / routine update; low urgency.",
        }
        self._department_labels = {
            "SAFETY": "Safety, security, passenger protection, emergency response.",
            "MAINTENANCE": "Repairs, mechanical/electrical faults, inspections, equipment maintenance.",
            "OPERATIONS": "Service operations, station/platform operations, scheduling, crowd management.",
            "COMPLIANCE": "Regulatory compliance, audit, policy violations, mandated procedures.",
            "FINANCE": "Payments, billing, penalties, budgeting, financial impact.",
        }
        self._alert_type_labels = {
            "safety_emergency": "Emergency, fire, evacuation, accident, injury, medical emergency.",
            "maintenance_critical": "Mechanical failure, brake/door malfunction, repair required, equipment breakdown.",
            "service_disruption": "Delays, cancellations, congestion, platform overcrowding, service disruption.",
            "infrastructure_issue": "Signal/track/power/infrastructure faults, outages, junction issues.",
            "security_concern": "Security threat, unauthorized access, vandalism, suspicious activity.",
            "regulatory_compliance": "Compliance issues, audit findings, regulatory violations, mandatory deadlines.",
            "financial_impact": "Fines, penalties, revenue/budget/billing issues.",
            "operational_standard": "Routine operations, standard procedure updates, scheduled work.",
        }

        # Pre-compute label embeddings once (fast inference)
        self._severity_matrix = self._encode_label_map(self._severity_labels)
        self._department_matrix = self._encode_label_map(self._department_labels)
        self._alert_type_matrix = self._encode_label_map(self._alert_type_labels)

        # ML keyword extraction (embedding-based)
        self.keyword_extractor = KeyBERT(model=self.embedder)

        # Confidence cache for alerts (text -> response)
        self.alert_cache = {}
        
        print("✓ Alert classification system ready (ML embeddings)")

    def _encode_label_map(self, label_map: Dict[str, str]) -> np.ndarray:
        labels = list(label_map.keys())
        descriptions = [label_map[k] for k in labels]
        emb = self.embedder.encode(descriptions, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def _predict_from_embeddings(self, text_embedding: np.ndarray, label_map: Dict[str, str], label_matrix: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        labels = list(label_map.keys())
        sims = np.dot(label_matrix, text_embedding)
        probs = self._softmax(sims * 10.0)  # temperature for sharper probabilities
        best_idx = int(np.argmax(probs))
        best_label = labels[best_idx]
        best_prob = float(probs[best_idx])
        score_map = {labels[i]: float(probs[i]) for i in range(len(labels))}
        return best_label, best_prob, score_map

    def _extract_keywords_ml(self, text: str, top_n: int = 5) -> List[str]:
        try:
            keywords = self.keyword_extractor.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words=None,
                top_n=top_n,
            )
            # returns list of tuples: (phrase, score)
            return [k[0] for k in keywords if k and k[0]]
        except Exception:
            return []
    
    def _init_semantic_search(self):
        """Initialize semantic search with built-in metro rail knowledge base"""
        
        # Built-in knowledge base for metro rail operations (no external files needed)
        self.built_in_knowledge = [
            "Emergency brake failure requires immediate safety response and maintenance action",
            "Signal malfunction causes service disruption and requires operations team intervention", 
            "Fire alarm activation triggers evacuation procedures and safety protocols",
            "Power outage affects train operations and requires electrical maintenance",
            "Track obstruction creates safety hazard and needs immediate clearance",
            "Door mechanism failure prevents proper train operation and passenger safety",
            "Platform overcrowding requires crowd management and service adjustments", 
            "Medical emergency on train needs immediate first aid and ambulance response",
            "Security threat requires immediate safety measures and police intervention",
            "Equipment malfunction disrupts normal operations and needs technical repair",
            "Passenger injury requires medical attention and incident documentation",
            "System failure affects multiple services and needs emergency response coordination",
            "Maintenance inspection ensures safe and reliable train operations",
            "Safety protocol compliance prevents accidents and ensures passenger security",
            "Technical malfunction requires expert diagnosis and professional repair",
            "Service delay impacts passenger schedules and requires clear communication",
            "Infrastructure damage needs structural assessment and specialized repair",
            "Communication system failure affects operational coordination and safety",
            "Environmental hazard requires safety evaluation and protective measures",
            "Regulatory violation needs immediate correction and compliance review",
            "Happy employees contribute to positive workplace culture and productivity",
            "Satisfied customers provide positive feedback about metro rail services",
            "Successful project completion brings joy and recognition to the team",
            "Cheerful staff improve passenger experience and service quality"
        ]
        
        # Document storage for both built-in and dynamic content
        self.documents = []
        self.document_vectors = []
        self.document_metadata = []
        
        # Index built-in knowledge base automatically
        for i, doc in enumerate(self.built_in_knowledge):
            self.documents.append(doc)
            vector = self._create_text_vector(doc)
            self.document_vectors.append(vector)
            self.document_metadata.append({
                "source": "built-in_knowledge",
                "index": i,
                "type": "metro_rail_operations"
            })
        
        # Search cache for performance
        self.search_cache = {}
        
        print(f"✓ Semantic search system ready with {len(self.built_in_knowledge)} built-in knowledge entries")
    
    # =========================================================================
    # ALERT CLASSIFICATION METHODS
    # =========================================================================
    
    def classify_alert(self, text: str) -> AlertResponse:
        """
        Classify alert text with enterprise-grade analysis
        
        Args:
            text: Alert text to classify
            
        Returns:
            Alert classification response in standardized format
        """
        start_time = time.time()
        
        self.metrics["alert_classifications"] += 1
        
        try:
            # Check cache
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self.alert_cache:
                cached_result = self.alert_cache[cache_key]
                self.metrics["cache_hits"] += 1
                return AlertResponse(**cached_result)
            
            # Create a true model-based embedding for the input
            text_vec = self._create_text_vector(text)

            # Predict each field using embedding similarity to label descriptions
            severity, severity_conf, _ = self._predict_from_embeddings(
                text_vec, self._severity_labels, self._severity_matrix
            )
            department, dept_conf, _ = self._predict_from_embeddings(
                text_vec, self._department_labels, self._department_matrix
            )
            alert_type, type_conf, _ = self._predict_from_embeddings(
                text_vec, self._alert_type_labels, self._alert_type_matrix
            )

            # ML-based keyword extraction
            search_keywords = self._extract_keywords_ml(text, top_n=5)
            
            processing_time = round((time.time() - start_time), 3)
            
            # Build response in requested format
            response_data = {
                "status": "success",
                "severity": severity.lower(),
                "alert_type": alert_type.lower(),
                "department": department.lower(),
                "keywords": search_keywords[:5],
                "confidence": round(float((severity_conf + dept_conf + type_conf) / 3.0), 2),
                "processing_time": processing_time
            }
            
            # Cache result
            if len(self.alert_cache) < 1000:
                self.alert_cache[cache_key] = response_data.copy()
            
            # Update metrics
            self.metrics["avg_alert_time"] = (
                (self.metrics["avg_alert_time"] * (self.metrics["alert_classifications"] - 1) + 
                 processing_time * 1000) / self.metrics["alert_classifications"]
            )
            
            return AlertResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Alert classification failed: {e}")
            # Return error response
            return AlertResponse(
                status="error",
                severity="unknown",
                alert_type="system_error",
                department="operations",
                keywords=[],
                confidence=0.0,
                processing_time=round((time.time() - start_time), 3)
            )
    
    
    
    # =========================================================================
    # SEMANTIC SEARCH METHODS
    # =========================================================================
    
    def index_documents(self, documents: List[str], metadata: List[Dict] = None) -> Dict[str, Any]:
        """
        Index documents for semantic search
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for documents
            
        Returns:
            Indexing results
        """
        start_time = time.time()
        print(f"\nIndexing {len(documents)} documents for semantic search...")
        
        # Store documents
        self.documents = documents
        self.document_metadata = metadata or [{"index": i} for i in range(len(documents))]
        
        # Create vectors for all documents
        self.document_vectors = []
        for i, doc in enumerate(documents):
            if i % 10 == 0:
                print(f"  Processing document {i+1}/{len(documents)}")
            
            vector = self._create_text_vector(doc)
            self.document_vectors.append(vector)
        
        indexing_time = (time.time() - start_time) * 1000
        self.metrics["total_indexed_docs"] = len(documents)
        
        result = {
            "status": "success",
            "indexed_documents": len(documents),
            "indexing_time_ms": round(indexing_time, 2),
            "vector_dimension": self.embedding_dim
        }
        
        print(f"✓ Indexing completed in {indexing_time:.1f}ms")
        return result
    
    def search_documents(self, query: str, top_k: int = 10, threshold: float = 0.1) -> SearchResponse:
        """
        Perform semantic search on indexed documents
        
        Args:
            query: Search query text
            top_k: Maximum results to return
            threshold: Similarity threshold
            
        Returns:
            Complete search response
        """
        start_time = time.time()
        self.metrics["semantic_searches"] += 1
        
        if not self.documents:
            return SearchResponse(
                query=query,
                timestamp=datetime.now().isoformat(),
                total_results=0,
                search_time_ms=0,
                results=[],
                expanded_terms=[],
                search_metadata={"error": "No documents indexed"}
            )
        
        try:
            # Check cache
            cache_key = hashlib.md5(f"{query}_{top_k}_{threshold}".encode()).hexdigest()
            if cache_key in self.search_cache:
                cached_response = self.search_cache[cache_key]
                self.metrics["cache_hits"] += 1
                return cached_response
            
            # Create query vector
            query_vector = self._create_text_vector(query)
            
            # Calculate similarities
            similarities = []
            for i, doc_vector in enumerate(self.document_vectors):
                if np.linalg.norm(doc_vector) > 0 and np.linalg.norm(query_vector) > 0:
                    similarity = np.dot(query_vector, doc_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
                    )
                else:
                    similarity = 0.0
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Format results
            results = []
            for rank, (doc_idx, sim_score) in enumerate(similarities[:top_k], 1):
                if sim_score >= threshold:
                    doc_content = self.documents[doc_idx]
                    snippet = doc_content[:100] + "..." if len(doc_content) > 100 else doc_content
                    
                    result = SearchResult(
                        document_id=f"doc_{doc_idx}_{hashlib.md5(doc_content.encode()).hexdigest()[:8]}",
                        content=doc_content,
                        similarity_score=round(sim_score, 4),
                        rank=rank,
                        snippet=snippet
                    )
                    results.append(result)
            
            # Generate expanded terms
            expanded_terms = self._get_expanded_terms(query)
            
            search_time = (time.time() - start_time) * 1000
            
            response = SearchResponse(
                query=query,
                timestamp=datetime.now().isoformat(),
                total_results=len(results),
                search_time_ms=round(search_time, 2),
                results=results,
                expanded_terms=expanded_terms,
                search_metadata={
                    "total_documents_searched": len(self.documents),
                    "similarity_threshold": threshold,
                    "search_algorithm": "cosine_similarity"
                }
            )
            
            # Cache result
            if len(self.search_cache) < 1000:
                self.search_cache[cache_key] = response
            
            # Update metrics
            self.metrics["avg_search_time"] = (
                (self.metrics["avg_search_time"] * (self.metrics["semantic_searches"] - 1) + 
                 search_time) / self.metrics["semantic_searches"]
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return SearchResponse(
                query=query,
                timestamp=datetime.now().isoformat(),
                total_results=0,
                search_time_ms=0,
                results=[],
                expanded_terms=[],
                search_metadata={"error": str(e)}
            )
    
    def _create_text_vector(self, text: str) -> np.ndarray:
        """Create a normalized sentence embedding using the transformer model."""
        try:
            vec = self.embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            vec = np.asarray(vec, dtype=np.float32)
            return vec
            
        except Exception as e:
            logger.error(f"Error creating vector: {e}")
            # Return a zero vector as fallback (keeps downstream logic predictable)
            return np.zeros((self.embedding_dim,), dtype=np.float32)
    
    def get_text_embedding(self, text: str) -> EmbeddingResponse:
        """
        Generate text embedding for API service
        
        Args:
            text: Input text to convert to embedding
            
        Returns:
            EmbeddingResponse with 512-dimensional embedding array
        """
        start_time = time.time()
        
        try:
            # Generate embedding using existing method
            embedding_array = self._create_text_vector(text)
            
            # Convert numpy array to Python list with full precision
            embedding_list = [float(x) for x in embedding_array.tolist()]
            
            # Validate embedding quality
            if all(abs(x) < 1e-8 for x in embedding_list):
                raise ValueError("Generated embedding is all zeros - possible tokenization issue")
            
            processing_time = round((time.time() - start_time), 3)
            
            response = EmbeddingResponse(
                status="success",
                embedding=embedding_list,
                model=self.embedding_model_name,
                dimension=int(embedding_array.shape[0]),
                processing_time=processing_time
            )
            
            # Validate response
            if not response.validate_embedding():
                raise ValueError("Embedding validation failed")
                
            return response
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            processing_time = round((time.time() - start_time), 3)
            
            return EmbeddingResponse(
                status="error",
                embedding=[0.0] * int(self.embedding_dim),
                model=self.embedding_model_name,
                dimension=int(self.embedding_dim),
                processing_time=processing_time
            )
    
    def _get_expanded_terms(self, query: str) -> List[str]:
        """Optional query expansion.

        For a purely ML-based pipeline, we don't inject hardcoded synonym maps.
        """
        return []
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get unified system performance metrics"""
        return {
            "system_info": {
                "version": "6.0-unified",
                "device": str(self.device),
                "embedding_model": self.embedding_model_name,
                "embedding_dim": int(self.embedding_dim)
            },
            "alert_classification": {
                "total_classifications": self.metrics["alert_classifications"],
                "avg_processing_time_ms": round(self.metrics["avg_alert_time"], 2),
                "cached_alerts": len(self.alert_cache)
            },
            "semantic_search": {
                "total_searches": self.metrics["semantic_searches"],
                "avg_search_time_ms": round(self.metrics["avg_search_time"], 2),
                "indexed_documents": self.metrics["total_indexed_docs"],
                "cached_searches": len(self.search_cache)
            },
            "overall_performance": {
                "total_cache_hits": self.metrics["cache_hits"],
                "cache_hit_rate": (self.metrics["cache_hits"] / max(1, 
                    self.metrics["alert_classifications"] + self.metrics["semantic_searches"])) * 100
            }
        }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_documents_from_file(filepath: str) -> List[str]:
    """Load documents from text file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]
        print(f"✓ Loaded {len(documents)} documents from {filepath}")
        return documents
    except Exception as e:
        print(f"Error loading documents from {filepath}: {e}")
        return []

def print_alert_result(response: AlertResponse):
    """Print alert classification result in JSON format only"""
    print(response.to_json())

def print_embedding_result(response: EmbeddingResponse):
    """Print text embedding result in JSON format only"""
    print(response.to_json())

def print_search_results(response: SearchResponse):
    """Print semantic search results in JSON format only"""
    print(response.to_json())
    
    for result in response.results:
        print(f"\n[{result.rank}] Similarity: {result.similarity_score:.3f}")
        print(f"Document: {result.content}")

def main():
    """Main function - starts interactive mode directly"""
    import sys
    
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print('''
KMRL Unified ML System - Interactive Mode
==========================================

Usage:
    python kmrl_unified_system.py
    
Commands (in interactive mode):
    classify <text>     - Classify alert text
    search <query>      - Search for similar documents
    index <file>        - Index documents from file  
    metrics             - Show system metrics
    help                - Show commands
    exit                - Exit program

Example:
    🤖 KMRL> index sample_documents.txt
    🤖 KMRL> search happy employees
    🤖 KMRL> classify Emergency brake failure
        ''')
        return
    
    # Initialize unified system
    try:
        system = KMRLUnifiedMLSystem()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Start interactive mode directly
    print("\n" + "="*80)
    print("KMRL UNIFIED ML SYSTEM - INTERACTIVE MODE")
    print("="*80)
    print("Available commands:")
    print("  classify <text>     - Classify alert text")
    print("  embed <text>        - Generate text embedding")
    print("  search <query>      - Search built-in knowledge base")
    print("  json <command>      - Get raw JSON output")
    print("  index <documents>   - Add custom documents (optional)")
    print("  metrics             - Show system performance")
    print("  help                - Show this help")
    print("  exit                - Exit the program")
    print("="*80)
    
    while True:
        try:
            user_input = input("\n🤖 KMRL> ").strip()
            
            if not user_input:
                continue
            elif user_input.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  classify <text>     - Classify alert text") 
                print("  embed <text>        - Generate text embedding")
                print("  search <query>      - Search knowledge base")
                print("  json <command>      - Get raw JSON output")
                print("  index <file>        - Index documents")
                print("  metrics             - Show metrics")
                print("  exit                - Exit program")
                continue
            elif user_input.lower() == 'metrics':
                metrics = system.get_system_metrics()
                print(json.dumps(metrics, indent=2))
                continue
            elif user_input.startswith('json '):
                # Handle raw JSON output commands
                json_cmd = user_input[5:].strip()
                if json_cmd.startswith('classify '):
                    text = json_cmd[9:].strip()
                    if text:
                        response = system.classify_alert(text)
                        print(response.to_json())
                    else:
                        print("Please provide text to classify")
                elif json_cmd.startswith('embed '):
                    text = json_cmd[6:].strip()
                    if text:
                        response = system.get_text_embedding(text)
                        print(response.to_json())
                    else:
                        print("Please provide text to embed")
                elif json_cmd.startswith('search '):
                    query = json_cmd[7:].strip()
                    if query:
                        response = system.search_documents(query, top_k=5)
                        print(response.to_json())
                    else:
                        print("Please provide search query")
                else:
                    print("Invalid json command. Use: json classify <text>, json embed <text>, or json search <query>")
            elif user_input.startswith('classify '):
                text = user_input[9:].strip()
                if text:
                    response = system.classify_alert(text)
                    print_alert_result(response)
                else:
                    print("Please provide text to classify")
            elif user_input.startswith('embed '):
                text = user_input[6:].strip()
                if text:
                    response = system.get_text_embedding(text)
                    print_embedding_result(response)
                else:
                    print("Please provide text to embed")
            elif user_input.startswith('search '):
                query = user_input[7:].strip()
                if query:
                    response = system.search_documents(query, top_k=5)
                    print_search_results(response)
                else:
                    print("Please provide search query")
            elif user_input.startswith('index '):
                filepath = user_input[6:].strip()
                if filepath:
                    documents = load_documents_from_file(filepath)
                    if documents:
                        result = system.index_documents(documents)
                        print(f"✓ Indexed {result['indexed_documents']} documents")
                else:
                    print("Please provide file path")
            else:
                print("Unknown command. Type 'help' for available commands.")
            
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# =============================================================================
# API SERVICE FUNCTIONALITY
# =============================================================================

def create_api_app(ml_system_instance=None):
    """
    Create Flask API application
    
    Args:
        ml_system_instance: Pre-initialized ML system instance
        
    Returns:
        Flask app with API routes
    """
    if not FLASK_AVAILABLE:
        raise ImportError("Flask not available. Install with: pip install flask flask-cors")
    
    app = Flask(__name__)
    CORS(app)  # Enable CORS for frontend integration
    
    # Use provided ML system or create new one
    if ml_system_instance is None:
        ml_system_instance = KMRLUnifiedMLSystem()
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "service": "KMRL ML API",
            "version": "6.0",
            "timestamp": datetime.now().isoformat()
        })

    @app.route('/classify', methods=['POST'])
    def classify_alert():
        """
        Alert classification endpoint
        
        Request: {"text": "alert message"}
        Response: {"status": "success", "severity": "high", ...}
        """
        try:
            # Validate request
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
                
            data = request.get_json()
            if 'text' not in data:
                return jsonify({"error": "Missing 'text' field in request"}), 400
                
            text = data['text'].strip()
            if not text:
                return jsonify({"error": "Text field cannot be empty"}), 400
            
            # Classify alert
            response = ml_system_instance.classify_alert(text)
            
            return jsonify(response.to_dict())
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": "Internal server error",
                "message": str(e)
            }), 500

    @app.route('/api/analyze', methods=['POST'])
    def analyze_alert():
        """Unified analysis endpoint for web UI.

        Request: {"text": "alert message"}
        Response: {severity, department, confidence, keywords, semantic_similarity_results, immediate_action_required, immediate_action}
        """
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400

            data = request.get_json()
            if 'text' not in data:
                return jsonify({"error": "Missing 'text' field in request"}), 400

            text = str(data['text']).strip()
            if not text:
                return jsonify({"error": "Text field cannot be empty"}), 400

            # 1) Classification
            classification = ml_system_instance.classify_alert(text)
            severity = str(classification.severity).strip().upper()
            department = str(classification.department).strip().upper()
            confidence = float(classification.confidence)
            keywords = list(classification.keywords or [])

            # 2) Semantic similarity (using built-in knowledge base/index)
            search = ml_system_instance.search_documents(text, top_k=5, threshold=0.1)
            semantic_similarity_results = [
                {
                    "text": r.content,
                    "similarity": float(r.similarity_score),
                }
                for r in (search.results or [])
            ]

            # 3) Immediate action heuristic
            immediate_action_required = severity in {"CRITICAL", "HIGH"}
            if immediate_action_required:
                immediate_action = f"Escalate immediately to {department} and follow the incident runbook."
            else:
                immediate_action = "Monitor and route to the owning team."

            return jsonify({
                "status": "success",
                "severity": severity,
                "department": department,
                "confidence": confidence,
                "keywords": keywords,
                "immediate_action_required": immediate_action_required,
                "immediate_action": immediate_action,
                "semantic_similarity_results": semantic_similarity_results,
            })

        except Exception as e:
            return jsonify({
                "status": "error",
                "error": "Internal server error",
                "message": str(e)
            }), 500

    @app.route('/embed', methods=['POST'])
    def embed_text():
        """
        Text embedding endpoint
        
        Request: {"text": "query text"}
        Response: {"status": "success", "embedding": [N values], "model": "...", "dimension": N, "processing_time": 0.001}
        """
        try:
            # Validate request
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
                
            data = request.get_json()
            if 'text' not in data:
                return jsonify({"error": "Missing 'text' field in request"}), 400
                
            text = data['text'].strip()
            if not text:
                return jsonify({"error": "Text field cannot be empty"}), 400
            
            # Generate embedding
            embed_response = ml_system_instance.get_text_embedding(text)
            
            return jsonify(embed_response.to_dict())
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": "Internal server error", 
                "message": str(e)
            }), 500

    @app.route('/metrics', methods=['GET'])
    def get_metrics():
        """Get system performance metrics"""
        try:
            metrics = ml_system_instance.get_system_metrics()
            return jsonify(metrics)
        except Exception as e:
            return jsonify({"error": "Failed to get metrics"}), 500

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        return jsonify({
            "error": "Not found",
            "message": "Endpoint not found. Available: /analyze, /classify, /embed, /health, /metrics"
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 errors"""
        return jsonify({
            "error": "Method not allowed",
            "message": "Check request method. Most endpoints require POST."
        }), 405
    
    return app

def start_api_server(host='0.0.0.0', port=8000, debug=False):
    """
    Start the API server
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    if not FLASK_AVAILABLE:
        print("❌ Flask not available. Install with: pip install flask flask-cors")
        return
    
    print("KMRL ML API Service")
    print("=" * 50)
    print("Available endpoints:")
    print("  POST /analyze     - Unified alert analysis (UI endpoint)")
    print("  POST /classify    - Alert classification")
    print("  POST /embed       - Text embedding (model-based)")
    print("  GET  /health      - Health check")
    print("  GET  /metrics     - System metrics")
    print("=" * 50)
    
    # Initialize ML system and create app
    ml_system = KMRLUnifiedMLSystem()
    app = create_api_app(ml_system)
    
    # Start server
    app.run(host=host, port=port, debug=debug, threaded=True)

def main():
    """Main entry point with mode selection"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--api':
            # API server mode
            start_api_server()
            return
        elif sys.argv[1] == '--help':
            print("KMRL Unified ML System")
            print("Usage:")
            print("  python kmrl_unified_system.py         - Interactive mode")
            print("  python kmrl_unified_system.py --api   - Start API server")
            print("  python kmrl_unified_system.py --help  - Show this help")
            return
    
    # Default: Interactive mode
    system = KMRLUnifiedMLSystem()
    
    # Start interactive mode directly
    while True:
        try:
            user_input = input("\n🤖 KMRL> ").strip()
            
            if not user_input:
                continue
            elif user_input.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  classify <text>     - Classify alert text") 
                print("  embed <text>        - Generate text embedding")
                print("  search <query>      - Search knowledge base")
                print("  json <command>      - Get raw JSON output")
                print("  index <file>        - Index documents")
                print("  metrics             - Show metrics")
                print("  exit                - Exit program")
                continue
            elif user_input.lower() == 'metrics':
                metrics = system.get_system_metrics()
                print(json.dumps(metrics, indent=2))
                continue
            elif user_input.startswith('json '):
                # Handle raw JSON output commands
                json_cmd = user_input[5:].strip()
                if json_cmd.startswith('classify '):
                    text = json_cmd[9:].strip()
                    if text:
                        response = system.classify_alert(text)
                        print(response.to_json())
                    else:
                        print("Please provide text to classify")
                elif json_cmd.startswith('embed '):
                    text = json_cmd[6:].strip()
                    if text:
                        response = system.get_text_embedding(text)
                        print(response.to_json())
                    else:
                        print("Please provide text to embed")
                elif json_cmd.startswith('search '):
                    query = json_cmd[7:].strip()
                    if query:
                        response = system.search_documents(query, top_k=5)
                        print(response.to_json())
                    else:
                        print("Please provide search query")
                else:
                    print("Invalid json command. Use: json classify <text>, json embed <text>, or json search <query>")
            elif user_input.startswith('classify '):
                text = user_input[9:].strip()
                if text:
                    response = system.classify_alert(text)
                    print_alert_result(response)
                else:
                    print("Please provide text to classify")
            elif user_input.startswith('embed '):
                text = user_input[6:].strip()
                if text:
                    response = system.get_text_embedding(text)
                    print_embedding_result(response)
                else:
                    print("Please provide text to embed")
            elif user_input.startswith('search '):
                query = user_input[7:].strip()
                if query:
                    response = system.search_documents(query, top_k=5)
                    print_search_results(response)
                else:
                    print("Please provide search query")
            elif user_input.startswith('index '):
                filepath = user_input[6:].strip()
                if filepath:
                    documents = load_documents_from_file(filepath)
                    if documents:
                        result = system.index_documents(documents)
                        print(f"✓ Indexed {result['indexed_documents']} documents")
                else:
                    print("Please provide file path")
            else:
                print("Unknown command. Type 'help' for available commands.")
            
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()