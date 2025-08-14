# hybrid_search_engine.py
"""
Part 1: Hybrid Search Engine & Base Classes
Core RAG implementation for AI Overview optimization
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from rerankers import Reranker
import logging
from datetime import datetime
from pathlib import Path

# Download required NLTK data
NLTK_DATA_DIR = "/tmp/nltk_data"
import os
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

try:
    nltk.data.find('tokenizers/punkt')
    print("✅ NLTK punkt tokenizer already available")
except LookupError:
    print("📦 Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=True)
    print("✅ NLTK punkt tokenizer downloaded")

try:
    nltk.data.find('tokenizers/punkt_tab')
    print("✅ NLTK punkt_tab tokenizer already available")
except LookupError:
    print("📦 Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab', download_dir=NLTK_DATA_DIR, quiet=True)
    print("✅ NLTK punkt_tab tokenizer downloaded")

try:
    nltk.data.find('corpora/stopwords')
    print("✅ NLTK stopwords already available")
except LookupError:
    print("📦 Downloading NLTK stopwords...")
    nltk.download('stopwords', download_dir=NLTK_DATA_DIR, quiet=True)
    print("✅ NLTK stopwords downloaded")

logger = logging.getLogger(__name__)

# Safe tokenization function with fallbacks
def safe_word_tokenize(text):
    """Safely tokenize text with fallbacks if NLTK resources are missing"""
    try:
        return word_tokenize(text.lower())
    except LookupError as e:
        # NLTK resource not found, use simple split
        print(f"⚠️ NLTK tokenization failed, using simple split: {e}")
        return text.lower().split()
    except Exception as e:
        # Any other error, use simple split
        print(f"⚠️ Tokenization error, using simple split: {e}")
        return text.lower().split()

# ================================
# Base State Management
# ================================

@dataclass
class AIOverviewState:
    """State object for AI Overview optimization workflow"""
    query: str = ""
    sub_intent: Dict = None
    candidate_documents: List[Dict] = None
    hybrid_search_results: List[Dict] = None
    reranked_results: List[Dict] = None
    ground_truth_summary: str = ""
    generated_summary: str = ""
    validation_metrics: Dict = None
    optimization_recommendations: List[Dict] = None
    iteration_count: int = 0
    agent_history: List[Dict] = None
    debug_logs: List[str] = None

    def __post_init__(self):
        if self.candidate_documents is None:
            self.candidate_documents = []
        if self.hybrid_search_results is None:
            self.hybrid_search_results = []
        if self.reranked_results is None:
            self.reranked_results = []
        if self.validation_metrics is None:
            self.validation_metrics = {}
        if self.optimization_recommendations is None:
            self.optimization_recommendations = []
        if self.agent_history is None:
            self.agent_history = []
        if self.debug_logs is None:
            self.debug_logs = []
    
    def log_debug(self, message: str):
        """Add debug message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        debug_msg = f"[{timestamp}] {message}"
        self.debug_logs.append(debug_msg)
        print(debug_msg)

# ================================
# Hybrid Search Implementation
# ================================

class HybridSearchEngine:
    """
    Hybrid search combining vector and keyword search
    This is the core RAG component that mimics Google's retrieval
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"🔧 Initializing HybridSearchEngine with model: {model_name}")
        
        try:
            self.embedding_model = SentenceTransformer(model_name)
            print(f"✅ Embedding model loaded: {model_name}")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            raise
        
        self.vector_index = None
        self.bm25_index = None
        self.documents = []
        self.document_embeddings = None
        self.is_built = False
        
    def build_index(self, documents: List[Dict]) -> bool:
        """
        Build both vector and BM25 indices from documents
        Returns True if successful, raises exception if fails
        """
        if not documents:
            raise ValueError("❌ Cannot build index: No documents provided")
        
        print(f"🏗️ Building hybrid index for {len(documents)} documents...")
        
        try:
            self.documents = documents
            
            # Extract text content for indexing
            texts = []
            for i, doc in enumerate(documents):
                content = doc.get('content', '')
                if not content:
                    print(f"⚠️ Document {i} has empty content, using title as fallback")
                    content = doc.get('title', f'Document {i}')
                texts.append(content)
            
            if not any(texts):
                raise ValueError("❌ All documents have empty content")
            
            # Build vector index
            print("🔍 Building vector embeddings...")
            self.document_embeddings = self.embedding_model.encode(
                texts, 
                normalize_embeddings=True,
                show_progress_bar=True
            ).astype('float32')
            print(f"✅ Vector embeddings created: {self.document_embeddings.shape}")
            
            # Build BM25 index
            print("📚 Building BM25 keyword index...")
            tokenized_docs = []
            for text in texts:
                tokens = safe_word_tokenize(text)
                tokenized_docs.append(tokens)
            
            self.bm25_index = BM25Okapi(tokenized_docs)
            print(f"✅ BM25 index built with {len(tokenized_docs)} documents")
            
            self.is_built = True
            print(f"🎉 Hybrid index successfully built for {len(documents)} documents")
            return True
            
        except Exception as e:
            print(f"❌ Index building failed: {e}")
            self.is_built = False
            raise
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.6) -> List[Dict]:
        """
        Perform hybrid search combining vector and keyword search
        alpha: weight for vector search (0.6 = 60% vector, 40% keyword)
        """
        if not self.is_built:
            raise RuntimeError("❌ Index not built. Call build_index() first.")
        
        if not query.strip():
            raise ValueError("❌ Query cannot be empty")
        
        print(f"🔍 Hybrid search for: '{query}' (k={k}, alpha={alpha})")
        
        try:
            # Vector search
            print("🎯 Performing vector search...")
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True).astype('float32')
            
            # Calculate cosine similarities
            similarities = np.dot(self.document_embeddings, query_embedding.T).flatten()
            vector_indices = np.argsort(similarities)[-k:][::-1]
            
            print(f"📊 Vector search: top similarity = {similarities[vector_indices[0]]:.3f}")
            
            # Keyword search
            print("🔤 Performing BM25 keyword search...")
            tokenized_query = safe_word_tokenize(query)
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            keyword_indices = np.argsort(bm25_scores)[-k:][::-1]
            
            print(f"📊 Keyword search: top BM25 score = {bm25_scores[keyword_indices[0]]:.3f}")
            
            # Combine results
            all_candidate_indices = list(set(vector_indices.tolist() + keyword_indices.tolist()))
            print(f"🔗 Combined candidates: {len(all_candidate_indices)} unique documents")
            
            # Calculate combined scores
            results = []
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            
            for idx in all_candidate_indices:
                if idx < len(self.documents):
                    # Normalize scores to 0-1 range
                    vector_score = float(similarities[idx])
                    keyword_score = float(bm25_scores[idx]) / max_bm25
                    
                    combined_score = alpha * vector_score + (1 - alpha) * keyword_score
                    
                    result = {
                        'document': self.documents[idx].copy(),
                        'index': idx,
                        'vector_score': vector_score,
                        'keyword_score': keyword_score,
                        'combined_score': combined_score
                    }
                    results.append(result)
            
            # Sort by combined score
            results.sort(key=lambda x: x['combined_score'], reverse=True)
            final_results = results[:k]
            
            print(f"✅ Hybrid search complete: returning {len(final_results)} results")
            for i, result in enumerate(final_results[:3]):
                print(f"  {i+1}. Score: {result['combined_score']:.3f} (V:{result['vector_score']:.3f}, K:{result['keyword_score']:.3f})")
            
            return final_results
            
        except Exception as e:
            print(f"❌ Hybrid search failed: {e}")
            raise

# ================================
# Document Processing Utilities
# ================================

class DocumentProcessor:
    """Utility class for processing and preparing documents"""
    
    @staticmethod
    def create_document_from_serp(serp_entry: Dict) -> List[Dict]:
        """Create documents from SERP entry data"""
        documents = []
        
        try:
            # Extract AI Overview sources
            if serp_entry.get("has_ai_overview") and "ai_overview" in serp_entry:
                ai_overview = serp_entry["ai_overview"]
                
                # Add AI Overview content as a document
                if ai_overview.get("content"):
                    documents.append({
                        'url': f"ai_overview_{serp_entry.get('keyword', 'unknown')}",
                        'title': f"AI Overview: {serp_entry.get('keyword', 'Unknown Query')}",
                        'content': ai_overview["content"],
                        'type': 'ai_overview',
                        'query': serp_entry.get('keyword', ''),
                        'cited_in_aio': True
                    })
                    print(f"📄 Created AI Overview document for: {serp_entry.get('keyword', 'Unknown')}")
                
                # Add cited sources
                for i, source in enumerate(ai_overview.get("sources", [])):
                    if source.get("url") and source.get("title"):
                        documents.append({
                            'url': source["url"],
                            'title': source["title"],
                            'content': source.get("snippet", source.get("title", "")),  # Use snippet if available
                            'type': 'cited_source',
                            'query': serp_entry.get('keyword', ''),
                            'cited_in_aio': True,
                            'citation_rank': i + 1
                        })
                        print(f"📄 Created cited source document: {source['title'][:50]}...")
            
            # Add organic results as documents
            for i, organic in enumerate(serp_entry.get("organic_results", [])[:5]):  # Top 5 organic
                if organic.get("link") and organic.get("title"):
                    documents.append({
                        'url': organic["link"],
                        'title': organic["title"],
                        'content': organic.get("snippet", organic.get("title", "")),
                        'type': 'organic_result',
                        'query': serp_entry.get('keyword', ''),
                        'cited_in_aio': False,
                        'organic_rank': i + 1
                    })
                    
            print(f"✅ Created {len(documents)} documents from SERP entry")
            return documents
            
        except Exception as e:
            print(f"❌ Failed to create documents from SERP entry: {e}")
            return []
    
    @staticmethod
    def build_ground_truth_dataset(workflow_state: Dict) -> List[Dict]:
        """Build ground truth dataset from workflow state"""
        print("🏗️ Building ground truth dataset from SERP data...")
        
        ground_truth_data = []
        
        try:
            serp_data = workflow_state.get("serp_data_enhanced", [])
            if not serp_data:
                raise ValueError("No SERP data found in workflow state")
            
            print(f"📊 Processing {len(serp_data)} SERP entries...")
            
            for serp_entry in serp_data:
                if serp_entry.get("has_ai_overview"):
                    documents = DocumentProcessor.create_document_from_serp(serp_entry)
                    
                    if documents:
                        # Create ground truth entry
                        gt_entry = {
                            "query": serp_entry.get("keyword", ""),
                            "ai_overview_content": serp_entry.get("ai_overview", {}).get("content", ""),
                            "documents": documents,
                            "has_answer_box": serp_entry.get("answer_box", False),
                            "organic_count": serp_entry.get("organic_count", 0),
                            "paa_count": serp_entry.get("paa_count", 0)
                        }
                        ground_truth_data.append(gt_entry)
                        print(f"✅ Added ground truth for: {serp_entry.get('keyword', 'Unknown')}")
            
            print(f"🎉 Ground truth dataset built: {len(ground_truth_data)} entries")
            return ground_truth_data
            
        except Exception as e:
            print(f"❌ Failed to build ground truth dataset: {e}")
            raise

# ================================
# Validation Metrics Calculator
# ================================

class ValidationMetrics:
    """Calculate validation metrics for Judge Agent reliability"""
    
    @staticmethod
    def calculate_recall_at_k(predicted_urls: List[str], ground_truth_urls: List[str], k: int = 5) -> float:
        """Calculate Recall@K metric"""
        if not ground_truth_urls:
            return 0.0
        
        predicted_set = set(predicted_urls[:k])
        ground_truth_set = set(ground_truth_urls)
        
        intersection = predicted_set.intersection(ground_truth_set)
        recall = len(intersection) / len(ground_truth_set)
        
        print(f"📊 Recall@{k}: {recall:.3f} ({len(intersection)}/{len(ground_truth_set)} URLs found)")
        return recall
    
    @staticmethod
    def calculate_mrr(predicted_urls: List[str], ground_truth_urls: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        if not ground_truth_urls:
            return 0.0
        
        ground_truth_set = set(ground_truth_urls)
        
        for i, url in enumerate(predicted_urls):
            if url in ground_truth_set:
                mrr = 1.0 / (i + 1)
                print(f"📊 MRR: {mrr:.3f} (first relevant result at position {i+1})")
                return mrr
        
        print("📊 MRR: 0.000 (no relevant results found)")
        return 0.0
    
    @staticmethod
    def calculate_semantic_similarity(text1: str, text2: str, embedding_model: SentenceTransformer) -> float:
        """Calculate semantic similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        try:
            embeddings = embedding_model.encode([text1, text2])
            similarity = float(np.dot(embeddings[0], embeddings[1]))
            print(f"📊 Semantic similarity: {similarity:.3f}")
            return similarity
        except Exception as e:
            print(f"❌ Semantic similarity calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_rouge_l_simple(prediction: str, reference: str) -> float:
        """Simple ROUGE-L calculation (longest common subsequence)"""
        if not prediction or not reference:
            return 0.0
        
        # Simple word-level LCS
        pred_words = prediction.lower().split()
        ref_words = reference.lower().split()
        
        # Dynamic programming for LCS
        m, n = len(pred_words), len(ref_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_words[i-1] == ref_words[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # ROUGE-L F1 score
        if m == 0 or n == 0:
            return 0.0
        
        precision = lcs_length / m
        recall = lcs_length / n
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        print(f"📊 ROUGE-L F1: {f1:.3f} (LCS: {lcs_length}/{min(m,n)} words)")
        return f1

# ================================
# Testing and Validation
# ================================

def test_hybrid_search_engine():
    """Test the hybrid search engine with sample data"""
    print("🧪 Testing HybridSearchEngine...")
    
    # Sample documents
    test_documents = [
        {
            'url': 'https://example.com/banking1',
            'title': 'Digital Banking Solutions',
            'content': 'Modern digital banking offers convenient online services for customers. Mobile apps provide easy access to account management and transactions.',
            'type': 'test'
        },
        {
            'url': 'https://example.com/banking2', 
            'title': 'Investment Banking Guide',
            'content': 'Investment banking helps companies raise capital through various financial instruments. Services include underwriting and advisory.',
            'type': 'test'
        },
        {
            'url': 'https://example.com/banking3',
            'title': 'Personal Banking Options',
            'content': 'Personal banking includes savings accounts, checking accounts, loans, and credit cards. Choose the right bank for your needs.',
            'type': 'test'
        }
    ]
    
    try:
        # Initialize engine
        engine = HybridSearchEngine()
        
        # Build index
        engine.build_index(test_documents)
        
        # Test search
        results = engine.hybrid_search("digital banking services", k=3)
        
        print(f"✅ Test passed: Found {len(results)} results")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['document']['title']} (score: {result['combined_score']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests
    test_hybrid_search_engine()
