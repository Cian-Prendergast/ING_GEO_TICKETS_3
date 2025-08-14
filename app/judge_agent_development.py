# judge_agent_development.py
"""
Part 2: Judge Agent Development & Training
This is the DEVELOPMENT phase - building and validating the Judge Agent
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
from rerankers import Reranker

from .hybrid_search_engine import (
    HybridSearchEngine, 
    AIOverviewState, 
    DocumentProcessor, 
    ValidationMetrics
)

logger = logging.getLogger(__name__)

# ================================
# Judge Agent Development Class
# ================================

class JudgeAgentDevelopment:
    """
    Development system for building and validating the Judge Agent
    This trains the agent to mimic Google's AI Overview selection
    """
    
    def __init__(self, azure_ai_client=None, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"🏗️ Initializing Judge Agent Development System")
        print(f"🤖 Reranker model: {reranker_model}")
        
        self.azure_client = azure_ai_client
        self.hybrid_search = HybridSearchEngine()
        
        try:
            self.reranker = Reranker(reranker_model, model_type="cross-encoder", verbose=0)
            print(f"✅ Reranker model loaded successfully")
        except Exception as e:
            print(f"⚠️ Reranker model loading failed: {e}")
            print("⚠️ Will continue without reranker (limited functionality)")
            self.reranker = None
        
        self.agent_id = f"judge_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.ground_truth_data = []
        self.validation_results = []
        self.is_trained = False
        
    def build_knowledge_base(self, workflow_state: Dict) -> bool:
        """
        Build knowledge base from ground truth AI Overview data
        This is the training data for the Judge Agent
        """
        print("🏗️ Building Judge Agent knowledge base...")
        
        try:
            # Extract ground truth from workflow state
            self.ground_truth_data = DocumentProcessor.build_ground_truth_dataset(workflow_state)
            
            if not self.ground_truth_data:
                raise ValueError("No ground truth data found - need AI Overview results first")
            
            # Flatten all documents for indexing
            all_documents = []
            for gt_entry in self.ground_truth_data:
                all_documents.extend(gt_entry.get('documents', []))
            
            if not all_documents:
                raise ValueError("No documents found in ground truth data")
            
            print(f"📚 Building hybrid search index with {len(all_documents)} documents...")
            success = self.hybrid_search.build_index(all_documents)
            
            if success:
                print(f"✅ Knowledge base built successfully")
                print(f"📊 Ground truth entries: {len(self.ground_truth_data)}")
                print(f"📊 Total documents: {len(all_documents)}")
                
                # Save ground truth data
                self._save_ground_truth_data()
                return True
            else:
                raise Exception("Hybrid search index building failed")
                
        except Exception as e:
            print(f"❌ Knowledge base building failed: {e}")
            raise
    
    def validate_judge_agent(self, test_queries: List[str] = None) -> Dict:
        """
        Validate the Judge Agent against ground truth data
        This measures how well it can mimic Google's AI Overview selection
        """
        if not self.hybrid_search.is_built:
            raise RuntimeError("❌ Knowledge base not built. Run build_knowledge_base() first.")
        
        print("🔍 Validating Judge Agent performance...")
        
        # Use test queries or extract from ground truth
        if test_queries is None:
            test_queries = [gt['query'] for gt in self.ground_truth_data if gt.get('query')]
        
        if not test_queries:
            raise ValueError("No test queries available")
        
        print(f"🧪 Testing with {len(test_queries)} queries...")
        
        validation_results = {
            'total_queries': len(test_queries),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'metrics': {
                'avg_recall_at_k': 0.0,
                'avg_mrr': 0.0,
                'avg_semantic_similarity': 0.0,
                'avg_rouge_l': 0.0,
                'avg_overall_score': 0.0
            },
            'query_results': [],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        total_metrics = {
            'recall_at_k': [],
            'mrr': [],
            'semantic_similarity': [],
            'rouge_l': [],
            'overall_score': []
        }
        
        for i, query in enumerate(test_queries):
            print(f"\n🔍 Analyzing query {i+1}/{len(test_queries)}: '{query}'")
            
            try:
                # Find ground truth for this query
                gt_entry = next((gt for gt in self.ground_truth_data if gt['query'] == query), None)
                if not gt_entry:
                    print(f"⚠️ No ground truth found for query: {query}")
                    continue
                
                # Run Judge analysis
                state = AIOverviewState(
                    query=query,
                    ground_truth_summary=gt_entry.get('ai_overview_content', ''),
                    iteration_count=1
                )
                
                analyzed_state = self._analyze_query(state)
                
                if analyzed_state.validation_metrics.get('error'):
                    print(f"❌ Analysis failed for query: {query}")
                    validation_results['failed_analyses'] += 1
                    continue
                
                # Store results
                query_result = {
                    'query': query,
                    'metrics': analyzed_state.validation_metrics,
                    'search_results_count': len(analyzed_state.hybrid_search_results),
                    'reranked_results_count': len(analyzed_state.reranked_results),
                    'generated_summary_length': len(analyzed_state.generated_summary) if analyzed_state.generated_summary else 0
                }
                
                validation_results['query_results'].append(query_result)
                validation_results['successful_analyses'] += 1
                
                # Accumulate metrics
                metrics = analyzed_state.validation_metrics
                for metric_key in total_metrics.keys():
                    if metric_key in metrics:
                        total_metrics[metric_key].append(metrics[metric_key])
                
                print(f"✅ Query analysis complete - Overall score: {metrics.get('overall_score', 0):.3f}")
                
            except Exception as e:
                print(f"❌ Failed to analyze query '{query}': {e}")
                validation_results['failed_analyses'] += 1
                continue
        
        # Calculate average metrics
        for metric_key, values in total_metrics.items():
            if values:
                avg_value = np.mean(values)
                validation_results['metrics'][f'avg_{metric_key}'] = float(avg_value)
                print(f"📊 Average {metric_key}: {avg_value:.3f}")
        
        # Determine if Judge Agent is reliable
        overall_avg = validation_results['metrics']['avg_overall_score']
        validation_results['is_reliable'] = overall_avg >= 0.6  # 60% threshold
        validation_results['reliability_level'] = self._get_reliability_level(overall_avg)
        
        print(f"\n🎯 Validation Summary:")
        print(f"   Successful analyses: {validation_results['successful_analyses']}/{validation_results['total_queries']}")
        print(f"   Average overall score: {overall_avg:.3f}")
        print(f"   Reliability level: {validation_results['reliability_level']}")
        print(f"   Judge Agent is {'✅ RELIABLE' if validation_results['is_reliable'] else '❌ NEEDS IMPROVEMENT'}")
        
        # Save validation results
        self.validation_results = validation_results
        self._save_validation_results()
        
        # Mark as trained if reliable
        if validation_results['is_reliable']:
            self.is_trained = True
            print(f"🎉 Judge Agent training completed successfully!")
        else:
            print(f"⚠️ Judge Agent needs improvement before production use")
        
        return validation_results
    
    def _analyze_query(self, state: AIOverviewState) -> AIOverviewState:
        """Internal method to analyze a single query"""
        state.log_debug(f"Starting Judge analysis for: {state.query}")
        
        try:
            # Step 1: Hybrid Search
            state.log_debug("Performing hybrid search...")
            search_results = self.hybrid_search.hybrid_search(
                state.query, 
                k=15,  # Get more candidates for reranking
                alpha=0.6
            )
            state.hybrid_search_results = search_results
            state.log_debug(f"Hybrid search found {len(search_results)} candidates")
            
            # Step 2: Reranking
            if search_results:
                state.log_debug("Reranking search results...")
                
                if self.reranker is not None:
                    docs_for_reranking = [
                        result['document']['content'][:1000]  # Truncate for efficiency
                        for result in search_results
                    ]
                    
                    ranked_results = self.reranker.rank(
                        query=state.query, 
                        docs=docs_for_reranking
                    )
                    
                    # Get top ranked documents
                    top_ranked = ranked_results.top_k(5)
                    state.reranked_results = [
                        {
                            'document': search_results[result.doc_id]['document'],
                            'rerank_score': result.score,
                            'original_search_score': search_results[result.doc_id]['combined_score'],
                            'rank_position': i + 1
                        }
                        for i, result in enumerate(top_ranked)
                    ]
                    state.log_debug(f"Reranking complete: {len(state.reranked_results)} top results")
                else:
                    # Fallback: use original search results without reranking
                    state.log_debug("Reranker not available - using original search order")
                    state.reranked_results = [
                        {
                            'document': result['document'],
                            'rerank_score': result['combined_score'],  # Use search score as rank score
                            'original_search_score': result['combined_score'],
                            'rank_position': i + 1
                        }
                        for i, result in enumerate(search_results[:5])
                    ]
            
            # Step 3: Content Generation (if Azure AI available)
            if state.reranked_results and self.azure_client:
                state.log_debug("Generating summary with Azure AI...")
                context_docs = [r['document'] for r in state.reranked_results[:3]]
                state.generated_summary = self._generate_summary(state.query, context_docs)
                state.log_debug(f"Summary generated: {len(state.generated_summary)} characters")
            
            # Step 4: Calculate Validation Metrics
            state.log_debug("Calculating validation metrics...")
            state.validation_metrics = self._calculate_validation_metrics(state)
            state.log_debug(f"Validation complete - Overall score: {state.validation_metrics.get('overall_score', 0):.3f}")
            
        except Exception as e:
            state.log_debug(f"Analysis failed: {e}")
            state.validation_metrics = {'error': str(e), 'overall_score': 0}
        
        return state
    
    def _generate_summary(self, query: str, context_docs: List[Dict]) -> str:
        """Generate AI Overview-style summary"""
        if not self.azure_client:
            return "Summary generation not available - Azure AI client not provided"
        
        context_text = "\n\n".join([
            f"Source: {doc.get('title', 'Unknown')}\nContent: {doc.get('content', '')[:500]}..."
            for doc in context_docs
        ])
        
        # Load prompt template
        prompt_path = Path("prompts/judge_summary_prompt.txt")
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                prompt_template = f.read()
            
            prompt = prompt_template.format(
                query=query,
                context_text=context_text
            )
        else:
            # Fallback prompt if file not found
            prompt = f"""
Generate a concise AI Overview summary for: "{query}"

Context sources:
{context_text}

Create a factual, well-structured summary that directly answers the query.
Summary:
"""
        
        try:
            # Call Azure AI (assuming it returns a dict with content)
            response = self.azure_client(prompt, max_tokens=300, temperature=0.1)
            if isinstance(response, dict):
                return response.get('choices', [{}])[0].get('message', {}).get('content', '')
            return str(response)
        except Exception as e:
            print(f"❌ Summary generation failed: {e}")
            return f"Summary generation failed: {str(e)}"
    
    def _calculate_validation_metrics(self, state: AIOverviewState) -> Dict:
        """Calculate comprehensive validation metrics"""
        metrics = {
            'recall_at_k': 0.0,
            'mrr': 0.0,
            'ndcg': 0.0,
            'semantic_similarity': 0.0,
            'rouge_l': 0.0,
            'overall_score': 0.0,
            'calculation_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Find ground truth for this query
            gt_entry = next(
                (gt for gt in self.ground_truth_data if gt['query'] == state.query), 
                None
            )
            
            if gt_entry:
                # Extract ground truth URLs
                gt_urls = [doc.get('url', '') for doc in gt_entry.get('documents', []) 
                          if doc.get('cited_in_aio', False)]
                
                # Extract predicted URLs
                predicted_urls = [r['document'].get('url', '') for r in state.reranked_results]
                
                # Calculate Recall@K
                metrics['recall_at_k'] = ValidationMetrics.calculate_recall_at_k(
                    predicted_urls, gt_urls, k=5
                )
                
                # Calculate MRR
                metrics['mrr'] = ValidationMetrics.calculate_mrr(predicted_urls, gt_urls)
                
                # Calculate semantic similarity
                if state.generated_summary and state.ground_truth_summary:
                    metrics['semantic_similarity'] = ValidationMetrics.calculate_semantic_similarity(
                        state.generated_summary, 
                        state.ground_truth_summary,
                        self.hybrid_search.embedding_model
                    )
                
                # Calculate ROUGE-L
                if state.generated_summary and state.ground_truth_summary:
                    metrics['rouge_l'] = ValidationMetrics.calculate_rouge_l_simple(
                        state.generated_summary, 
                        state.ground_truth_summary
                    )
            
            # Calculate overall score (weighted combination)
            metrics['overall_score'] = (
                metrics['recall_at_k'] * 0.3 +
                metrics['mrr'] * 0.2 +
                metrics['semantic_similarity'] * 0.3 +
                metrics['rouge_l'] * 0.2
            )
            
        except Exception as e:
            print(f"❌ Metrics calculation failed: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _get_reliability_level(self, score: float) -> str:
        """Determine reliability level based on score"""
        if score >= 0.8:
            return "EXCELLENT"
        elif score >= 0.6:
            return "GOOD" 
        elif score >= 0.4:
            return "FAIR"
        else:
            return "POOR"
    
    def _save_ground_truth_data(self):
        """Save ground truth data to file"""
        try:
            save_path = Path("data/judge_training/ground_truth_data.json")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump({
                    'agent_id': self.agent_id,
                    'creation_timestamp': datetime.now().isoformat(),
                    'ground_truth_count': len(self.ground_truth_data),
                    'data': self.ground_truth_data
                }, f, indent=2)
            
            print(f"💾 Ground truth data saved to: {save_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save ground truth data: {e}")
    
    def _save_validation_results(self):
        """Save validation results to file"""
        try:
            save_path = Path(f"data/judge_training/validation_results_{self.agent_id}.json")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            
            print(f"💾 Validation results saved to: {save_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save validation results: {e}")
    
    def export_training_report(self) -> Dict:
        """Export comprehensive training report"""
        if not self.validation_results:
            raise RuntimeError("No validation results available. Run validate_judge_agent() first.")
        
        report = {
            'judge_agent_id': self.agent_id,
            'training_completion_time': datetime.now().isoformat(),
            'is_production_ready': self.is_trained,
            'ground_truth_stats': {
                'total_queries': len(self.ground_truth_data),
                'total_documents': sum(len(gt.get('documents', [])) for gt in self.ground_truth_data)
            },
            'validation_summary': self.validation_results,
            'recommendations': self._generate_recommendations(),
            'next_steps': self._get_next_steps()
        }
        
        # Save report
        try:
            save_path = Path(f"data/judge_training/training_report_{self.agent_id}.json")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"📊 Training report saved to: {save_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save training report: {e}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if not self.validation_results:
            return ["Run validation first"]
        
        metrics = self.validation_results.get('metrics', {})
        
        if metrics.get('avg_recall_at_k', 0) < 0.5:
            recommendations.append("Improve hybrid search relevance - consider adjusting alpha parameter")
        
        if metrics.get('avg_semantic_similarity', 0) < 0.6:
            recommendations.append("Enhance content generation - summary quality needs improvement")
        
        if metrics.get('avg_mrr', 0) < 0.4:
            recommendations.append("Optimize reranking model - top results not relevant enough")
        
        if self.validation_results.get('failed_analyses', 0) > 0:
            recommendations.append("Investigate failed analyses - improve error handling")
        
        if not recommendations:
            recommendations.append("Judge Agent performing well - ready for production use")
        
        return recommendations
    
    def _get_next_steps(self) -> List[str]:
        """Get next steps based on training results"""
        if self.is_trained:
            return [
                "Deploy Judge Agent to production",
                "Integrate with Optimizer Agent",
                "Start content optimization workflow",
                "Monitor production performance"
            ]
        else:
            return [
                "Collect more ground truth data",
                "Tune hyperparameters (alpha, k values)",
                "Try different reranker models", 
                "Improve content generation prompts",
                "Re-run validation after improvements"
            ]

# ================================
# Development Workflow Functions
# ================================

def run_judge_development_workflow(workflow_state: Dict, azure_ai_client=None) -> Dict:
    """
    Complete Judge Agent development workflow
    This is the main function to call for development phase
    """
    print("🚀 Starting Judge Agent Development Workflow")
    print("=" * 60)
    
    try:
        # Initialize development system
        judge_dev = JudgeAgentDevelopment(azure_ai_client)
        
        # Step 1: Build knowledge base
        print("\n📚 STEP 1: Building Knowledge Base")
        print("-" * 40)
        judge_dev.build_knowledge_base(workflow_state)
        
        # Step 2: Validate Judge Agent
        print("\n🔍 STEP 2: Validating Judge Agent")
        print("-" * 40)
        validation_results = judge_dev.validate_judge_agent()
        
        # Step 3: Generate training report
        print("\n📊 STEP 3: Generating Training Report")
        print("-" * 40)
        training_report = judge_dev.export_training_report()
        
        print("\n🎉 Judge Agent Development Complete!")
        print("=" * 60)
        
        return {
            'development_status': 'complete',
            'judge_agent_id': judge_dev.agent_id,
            'is_production_ready': judge_dev.is_trained,
            'validation_results': validation_results,
            'training_report': training_report,
            'completion_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Judge Agent Development Failed: {e}")
        raise

# ================================
# Testing Function
# ================================

def test_judge_development():
    """Test Judge development with mock data"""
    print("🧪 Testing Judge Agent Development...")
    
    # Create mock workflow state
    mock_workflow_state = {
        "serp_data_enhanced": [
            {
                "keyword": "digital banking",
                "has_ai_overview": True,
                "ai_overview": {
                    "content": "Digital banking refers to online banking services that allow customers to manage their finances electronically.",
                    "sources": [
                        {
                            "url": "https://example.com/digital-banking",
                            "title": "Digital Banking Guide",
                            "snippet": "Comprehensive guide to digital banking services"
                        }
                    ]
                },
                "organic_results": [
                    {
                        "link": "https://bank.com/services",
                        "title": "Banking Services Online",
                        "snippet": "Access your bank account online 24/7"
                    }
                ]
            }
        ]
    }
    
    try:
        result = run_judge_development_workflow(mock_workflow_state)
        print("✅ Judge development test passed!")
        return result
    except Exception as e:
        print(f"❌ Judge development test failed: {e}")
        return None

if __name__ == "__main__":
    # Run test
    test_judge_development()
