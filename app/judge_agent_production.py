# judge_agent_production.py
"""
Part 3: Judge Agent Production System
This is the PRODUCTION phase - using the trained Judge Agent for content optimization
"""

import json
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import logging
from rerankers import Reranker

from .hybrid_search_engine import (
    HybridSearchEngine, 
    AIOverviewState, 
    ValidationMetrics
)

logger = logging.getLogger(__name__)

# ================================
# Production Judge Agent
# ================================

class JudgeAgentProduction:
    """
    Production Judge Agent for content optimization
    Uses trained models to evaluate content for AI Overview inclusion
    """
    
    def __init__(self, azure_ai_client=None, judge_agent_id: str = None):
        print(f"🤖 Initializing Production Judge Agent")
        
        self.azure_client = azure_ai_client
        self.hybrid_search = HybridSearchEngine()
        self.reranker = None
        self.agent_id = judge_agent_id or f"judge_prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.is_loaded = False
        self.training_data = None
        
        print(f"🆔 Judge Agent ID: {self.agent_id}")
    
    def load_trained_model(self, training_data_path: str = None) -> bool:
        """
        Load trained Judge Agent model and knowledge base
        """
        print("📂 Loading trained Judge Agent model...")
        
        try:
            # Load training data
            if training_data_path:
                data_path = Path(training_data_path)
            else:
                data_path = Path("data/judge_training/ground_truth_data.json")
            
            if not data_path.exists():
                raise FileNotFoundError(f"Training data not found at: {data_path}")
            
            with open(data_path, 'r') as f:
                training_file = json.load(f)
                self.training_data = training_file.get('data', [])
            
            print(f"✅ Training data loaded: {len(self.training_data)} entries")
            
            # Build knowledge base from training data
            all_documents = []
            for entry in self.training_data:
                all_documents.extend(entry.get('documents', []))
            
            if not all_documents:
                raise ValueError("No documents found in training data")
            
            # Build hybrid search index
            success = self.hybrid_search.build_index(all_documents)
            if not success:
                raise Exception("Failed to build hybrid search index")
            
            # Initialize reranker
            self.reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2", model_type="cross-encoder", verbose=0)
            print("✅ Reranker model loaded")
            
            self.is_loaded = True
            print("🎉 Judge Agent loaded successfully and ready for production!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Judge Agent: {e}")
            self.is_loaded = False
            raise
    
    def evaluate_content_for_aio(self, content: str, target_query: str, sub_intent: Dict = None) -> Dict:
        """
        Evaluate content for AI Overview inclusion probability
        This is the main production function
        """
        if not self.is_loaded:
            raise RuntimeError("❌ Judge Agent not loaded. Call load_trained_model() first.")
        
        print(f"🔍 Evaluating content for query: '{target_query}'")
        print(f"📄 Content length: {len(content)} characters")
        
        try:
            # Create evaluation state
            state = AIOverviewState(
                query=target_query,
                sub_intent=sub_intent or {},
                iteration_count=1
            )
            
            state.log_debug(f"Starting content evaluation for: {target_query}")
            
            # Step 1: Add the content as a candidate document
            candidate_doc = {
                'url': 'candidate_content',
                'title': f'Content for: {target_query}',
                'content': content,
                'type': 'candidate',
                'is_candidate': True
            }
            
            # Step 2: Perform hybrid search to get competing content
            state.log_debug("Performing hybrid search for competing content...")
            competing_results = self.hybrid_search.hybrid_search(target_query, k=10, alpha=0.6)
            
            # Add candidate content to results for comparison
            candidate_result = {
                'document': candidate_doc,
                'index': -1,  # Special index for candidate
                'vector_score': 0.0,
                'keyword_score': 0.0,
                'combined_score': 0.0
            }
            
            # Calculate candidate content scores
            if competing_results:
                # Use embedding model to calculate similarity
                candidate_embedding = self.hybrid_search.embedding_model.encode([content])
                query_embedding = self.hybrid_search.embedding_model.encode([target_query])
                
                vector_similarity = float(np.dot(candidate_embedding[0], query_embedding[0]))
                candidate_result['vector_score'] = vector_similarity
                candidate_result['combined_score'] = vector_similarity  # Simplified for candidate
            
            # Combine candidate with competing results
            all_results = [candidate_result] + competing_results[:9]  # Top 9 competitors + candidate
            state.hybrid_search_results = all_results
            
            state.log_debug(f"Hybrid search complete: {len(all_results)} documents for comparison")
            
            # Step 3: Rerank all content including candidate
            state.log_debug("Reranking content including candidate...")
            docs_for_reranking = [result['document']['content'][:1000] for result in all_results]
            
            ranked_results = self.reranker.rank(query=target_query, docs=docs_for_reranking)
            top_ranked = ranked_results.top_k(5)
            
            # Find candidate position in rankings
            candidate_rank = None
            candidate_score = 0.0
            
            reranked_list = []
            for i, result in enumerate(top_ranked):
                doc_data = all_results[result.doc_id]
                rerank_data = {
                    'document': doc_data['document'],
                    'rerank_score': result.score,
                    'original_search_score': doc_data['combined_score'],
                    'rank_position': i + 1,
                    'is_candidate': doc_data['document'].get('is_candidate', False)
                }
                
                if doc_data['document'].get('is_candidate', False):
                    candidate_rank = i + 1
                    candidate_score = result.score
                    state.log_debug(f"Candidate content ranked at position: {candidate_rank}")
                
                reranked_list.append(rerank_data)
            
            state.reranked_results = reranked_list
            
            # Step 4: Calculate inclusion probability
            inclusion_metrics = self._calculate_inclusion_probability(
                candidate_rank, 
                candidate_score, 
                state,
                content,
                target_query
            )
            
            state.validation_metrics = inclusion_metrics
            
            # Step 5: Generate recommendations if content doesn't rank well
            recommendations = self._generate_improvement_recommendations(
                candidate_rank,
                candidate_score,
                competing_results[:3],  # Top 3 competitors
                content,
                target_query
            )
            
            evaluation_result = {
                'query': target_query,
                'content_length': len(content),
                'candidate_rank': candidate_rank,
                'candidate_score': candidate_score,
                'inclusion_probability': inclusion_metrics.get('inclusion_probability', 0),
                'confidence_level': inclusion_metrics.get('confidence_level', 'low'),
                'competing_content_count': len(competing_results),
                'top_competitors': [
                    {
                        'title': r['document'].get('title', 'Unknown'),
                        'score': r.get('rerank_score', 0),
                        'rank': r.get('rank_position', 0)
                    }
                    for r in reranked_list[:3] if not r.get('is_candidate', False)
                ],
                'improvement_recommendations': recommendations,
                'detailed_metrics': inclusion_metrics,
                'evaluation_timestamp': datetime.now().isoformat(),
                'judge_agent_id': self.agent_id
            }
            
            state.log_debug(f"Evaluation complete - Inclusion probability: {inclusion_metrics.get('inclusion_probability', 0):.1f}%")
            
            return evaluation_result
            
        except Exception as e:
            print(f"❌ Content evaluation failed: {e}")
            return {
                'error': str(e),
                'inclusion_probability': 0,
                'confidence_level': 'error',
                'evaluation_timestamp': datetime.now().isoformat()
            }
    
    def _calculate_inclusion_probability(self, candidate_rank: Optional[int], candidate_score: float, 
                                       state: AIOverviewState, content: str, query: str) -> Dict:
        """Calculate AI Overview inclusion probability based on ranking performance"""
        
        metrics = {
            'inclusion_probability': 0.0,
            'confidence_level': 'low',
            'ranking_score': 0.0,
            'content_quality_score': 0.0,
            'relevance_score': 0.0,
            'calculation_details': {}
        }
        
        try:
            # Base probability from ranking position
            if candidate_rank is None:
                ranking_probability = 0.0
                metrics['calculation_details']['ranking'] = "Content not in top 5"
            elif candidate_rank == 1:
                ranking_probability = 85.0
                metrics['calculation_details']['ranking'] = "Ranked #1 - Excellent"
            elif candidate_rank == 2:
                ranking_probability = 70.0
                metrics['calculation_details']['ranking'] = "Ranked #2 - Very Good"
            elif candidate_rank == 3:
                ranking_probability = 55.0
                metrics['calculation_details']['ranking'] = "Ranked #3 - Good"
            elif candidate_rank == 4:
                ranking_probability = 35.0
                metrics['calculation_details']['ranking'] = "Ranked #4 - Fair"
            else:
                ranking_probability = 15.0
                metrics['calculation_details']['ranking'] = "Ranked #5 - Low"
            
            metrics['ranking_score'] = ranking_probability
            
            # Content quality adjustments
            content_length = len(content)
            if content_length < 200:
                length_penalty = -15.0
                metrics['calculation_details']['length'] = "Too short - penalty applied"
            elif content_length > 2000:
                length_penalty = -5.0
                metrics['calculation_details']['length'] = "Very long - minor penalty"
            else:
                length_penalty = 0.0
                metrics['calculation_details']['length'] = "Appropriate length"
            
            # Reranker score adjustment
            score_boost = min(20.0, candidate_score * 30)  # Up to 20% boost from reranker score
            metrics['calculation_details']['reranker_boost'] = f"+{score_boost:.1f}% from reranker score"
            
            # Sub-intent alignment (if available)
            intent_boost = 0.0
            if state.sub_intent and state.sub_intent.get('sub_intent'):
                sub_intent_text = state.sub_intent.get('sub_intent', '').lower()
                if sub_intent_text in content.lower():
                    intent_boost = 10.0
                    metrics['calculation_details']['sub_intent'] = "Sub-intent alignment bonus"
                else:
                    metrics['calculation_details']['sub_intent'] = "Sub-intent not well aligned"
            
            # Calculate final probability
            final_probability = max(0.0, min(100.0, 
                ranking_probability + length_penalty + score_boost + intent_boost
            ))
            
            metrics['inclusion_probability'] = final_probability
            metrics['content_quality_score'] = max(0, 50 + length_penalty + intent_boost)
            metrics['relevance_score'] = min(100, score_boost * 2)
            
            # Determine confidence level
            if final_probability >= 70:
                metrics['confidence_level'] = 'high'
            elif final_probability >= 40:
                metrics['confidence_level'] = 'medium'
            else:
                metrics['confidence_level'] = 'low'
            
            print(f"📊 Inclusion Probability Calculation:")
            print(f"   Base ranking: {ranking_probability:.1f}%")
            print(f"   Length adjustment: {length_penalty:.1f}%")
            print(f"   Reranker boost: +{score_boost:.1f}%")
            print(f"   Sub-intent bonus: +{intent_boost:.1f}%")
            print(f"   Final probability: {final_probability:.1f}%")
            print(f"   Confidence: {metrics['confidence_level'].upper()}")
            
        except Exception as e:
            print(f"❌ Probability calculation failed: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _generate_improvement_recommendations(self, candidate_rank: Optional[int], candidate_score: float,
                                            top_competitors: List[Dict], content: str, query: str) -> List[Dict]:
        """Generate specific recommendations to improve AI Overview inclusion chances"""
        
        recommendations = []
        
        try:
            # Ranking-based recommendations
            if candidate_rank is None or candidate_rank > 3:
                recommendations.append({
                    'category': 'content_relevance',
                    'priority': 'high',
                    'issue': 'Content not ranking high enough for AI Overview inclusion',
                    'recommendation': 'Improve content relevance and comprehensiveness',
                    'specific_actions': [
                        f'Add more specific information about "{query}"',
                        'Include relevant keywords naturally throughout content',
                        'Structure content with clear headings and sections'
                    ],
                    'expected_impact': '+15-25% inclusion probability'
                })
            
            # Content length recommendations
            content_length = len(content)
            if content_length < 200:
                recommendations.append({
                    'category': 'content_completeness',
                    'priority': 'high',
                    'issue': 'Content too brief for comprehensive AI Overview',
                    'recommendation': 'Expand content to provide more comprehensive coverage',
                    'specific_actions': [
                        'Add detailed explanations and examples',
                        'Include step-by-step instructions if applicable',
                        'Address related questions users might have'
                    ],
                    'expected_impact': '+20-30% inclusion probability'
                })
            
            # Competitor analysis recommendations
            if top_competitors:
                competitor_titles = [comp['document'].get('title', 'Unknown') for comp in top_competitors]
                recommendations.append({
                    'category': 'competitive_analysis',
                    'priority': 'medium',
                    'issue': 'Strong competition from established content',
                    'recommendation': 'Analyze and outperform top competitors',
                    'specific_actions': [
                        f'Review competitor content: {", ".join(competitor_titles[:2])}',
                        'Identify content gaps in competitor coverage',
                        'Add unique insights and value propositions'
                    ],
                    'expected_impact': '+10-20% inclusion probability'
                })
            
            # Low reranker score recommendations
            if candidate_score < 0.5:
                recommendations.append({
                    'category': 'content_quality',
                    'priority': 'high',
                    'issue': 'Content quality score below optimal threshold',
                    'recommendation': 'Improve content structure and clarity',
                    'specific_actions': [
                        'Use clear, concise language',
                        'Add bullet points and numbered lists',
                        'Include relevant examples and use cases',
                        'Ensure content directly answers the query'
                    ],
                    'expected_impact': '+15-25% inclusion probability'
                })
            
            # Structure recommendations
            if '<h2>' not in content and '<h3>' not in content:
                recommendations.append({
                    'category': 'content_structure',
                    'priority': 'medium',
                    'issue': 'Content lacks clear hierarchical structure',
                    'recommendation': 'Add clear headings and subheadings',
                    'specific_actions': [
                        'Use H2 and H3 tags for section organization',
                        'Create logical content flow',
                        'Add FAQ section if appropriate'
                    ],
                    'expected_impact': '+10-15% inclusion probability'
                })
            
            # If no major issues found
            if not recommendations:
                recommendations.append({
                    'category': 'optimization',
                    'priority': 'low',
                    'issue': 'Content performing well overall',
                    'recommendation': 'Fine-tune for maximum AI Overview inclusion',
                    'specific_actions': [
                        'Add schema markup for better content understanding',
                        'Include related keywords and synonyms',
                        'Optimize for featured snippet format'
                    ],
                    'expected_impact': '+5-10% inclusion probability'
                })
            
            print(f"💡 Generated {len(recommendations)} improvement recommendations")
            
        except Exception as e:
            print(f"❌ Failed to generate recommendations: {e}")
            recommendations.append({
                'category': 'error',
                'priority': 'high',
                'issue': f'Recommendation generation failed: {str(e)}',
                'recommendation': 'Manual content review required',
                'specific_actions': ['Review content manually', 'Check for technical issues'],
                'expected_impact': 'Unknown'
            })
        
        return recommendations
    
    def batch_evaluate_content(self, content_list: List[Dict]) -> List[Dict]:
        """
        Evaluate multiple pieces of content in batch
        content_list: List of {'content': str, 'query': str, 'sub_intent': Dict}
        """
        if not self.is_loaded:
            raise RuntimeError("❌ Judge Agent not loaded. Call load_trained_model() first.")
        
        print(f"📊 Batch evaluating {len(content_list)} pieces of content...")
        
        results = []
        for i, content_item in enumerate(content_list):
            print(f"\n📄 Evaluating content {i+1}/{len(content_list)}")
            
            try:
                result = self.evaluate_content_for_aio(
                    content=content_item.get('content', ''),
                    target_query=content_item.get('query', ''),
                    sub_intent=content_item.get('sub_intent', {})
                )
                result['batch_index'] = i
                results.append(result)
                
                print(f"✅ Content {i+1} evaluated - Probability: {result.get('inclusion_probability', 0):.1f}%")
                
            except Exception as e:
                print(f"❌ Failed to evaluate content {i+1}: {e}")
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'inclusion_probability': 0,
                    'evaluation_timestamp': datetime.now().isoformat()
                })
        
        print(f"\n🎉 Batch evaluation complete: {len(results)} results")
        return results

# ================================
# Production Utility Functions
# ================================

def load_production_judge_agent(azure_ai_client=None, training_data_path: str = None) -> JudgeAgentProduction:
    """
    Convenience function to load a production-ready Judge Agent
    """
    print("🚀 Loading Production Judge Agent...")
    
    try:
        judge = JudgeAgentProduction(azure_ai_client)
        judge.load_trained_model(training_data_path)
        return judge
    except Exception as e:
        print(f"❌ Failed to load production Judge Agent: {e}")
        raise

def quick_content_evaluation(content: str, query: str, azure_ai_client=None) -> Dict:
    """
    Quick evaluation function for single pieces of content
    """
    try:
        judge = load_production_judge_agent(azure_ai_client)
        return judge.evaluate_content_for_aio(content, query)
    except Exception as e:
        print(f"❌ Quick evaluation failed: {e}")
        return {
            'error': str(e),
            'inclusion_probability': 0,
            'confidence_level': 'error'
        }

# ================================
# Testing Function
# ================================

def test_production_judge():
    """Test production Judge Agent with sample content"""
    print("🧪 Testing Production Judge Agent...")
    
    # Sample content for testing
    test_content = """
    # Digital Banking Solutions
    
    Digital banking has revolutionized how customers interact with their financial institutions. 
    Modern digital banking platforms offer comprehensive online services that allow customers to:
    
    ## Key Features
    - 24/7 account access through mobile apps
    - Instant money transfers and payments
    - Real-time transaction notifications
    - Digital wallet integration
    
    ## Benefits for Customers
    Digital banking provides convenience, security, and accessibility. Customers can manage 
    their finances from anywhere, at any time, without visiting physical branches.
    
    ## Security Measures
    Advanced encryption and multi-factor authentication ensure customer data protection.
    """
    
    test_query = "digital banking services"
    
    try:
        # Note: This will fail without actual training data, but shows the interface
        result = quick_content_evaluation(test_content, test_query)
        print("✅ Production Judge test completed!")
        print(f"📊 Inclusion probability: {result.get('inclusion_probability', 0):.1f}%")
        return result
    except Exception as e:
        print(f"❌ Production Judge test failed (expected without training data): {e}")
        return None

if __name__ == "__main__":
    # Run test
    test_production_judge()