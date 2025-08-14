# optimizer_agent.py
"""
Part 4: Optimizer Agent
Generates specific content optimization recommendations based on Judge Agent analysis
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

from .hybrid_search_engine import AIOverviewState
from .judge_agent_production import JudgeAgentProduction

logger = logging.getLogger(__name__)

# ================================
# Optimizer Agent Class
# ================================

class OptimizerAgent:
    """
    Optimizer Agent that generates actionable content recommendations
    Uses Judge Agent analysis to provide specific optimization strategies
    """
    
    def __init__(self, azure_ai_client=None, judge_agent: JudgeAgentProduction = None):
        print(f"🎯 Initializing Optimizer Agent")
        
        self.azure_client = azure_ai_client
        self.judge_agent = judge_agent
        self.agent_id = f"optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.optimization_history = []
        
        print(f"🆔 Optimizer Agent ID: {self.agent_id}")
    
    def optimize_content_for_aio(self, content: str, target_query: str, sub_intent: Dict = None, 
                                competitor_analysis: Dict = None) -> Dict:
        """
        Main optimization function - analyzes content and provides specific recommendations
        """
        print(f"🎯 Optimizing content for query: '{target_query}'")
        print(f"📄 Original content length: {len(content)} characters")
        
        try:
            optimization_state = {
                'original_content': content,
                'target_query': target_query,
                'sub_intent': sub_intent or {},
                'competitor_analysis': competitor_analysis or {},
                'optimization_timestamp': datetime.now().isoformat(),
                'agent_id': self.agent_id
            }
            
            # Step 1: Get Judge Agent evaluation of current content
            print("🔍 Getting Judge Agent evaluation...")
            if self.judge_agent:
                judge_evaluation = self.judge_agent.evaluate_content_for_aio(content, target_query, sub_intent)
            else:
                print("⚠️ No Judge Agent available - using simplified analysis")
                judge_evaluation = self._simplified_content_analysis(content, target_query)
            
            optimization_state['judge_evaluation'] = judge_evaluation
            
            current_probability = judge_evaluation.get('inclusion_probability', 0)
            print(f"📊 Current AI Overview inclusion probability: {current_probability:.1f}%")
            
            # Step 2: Analyze content gaps and opportunities
            print("🔍 Analyzing content gaps...")
            gap_analysis = self._analyze_content_gaps(content, target_query, sub_intent, judge_evaluation)
            optimization_state['gap_analysis'] = gap_analysis
            
            # Step 3: Generate specific optimization recommendations
            print("💡 Generating optimization recommendations...")
            recommendations = self._generate_optimization_recommendations(
                content, target_query, sub_intent, judge_evaluation, gap_analysis, competitor_analysis
            )
            optimization_state['recommendations'] = recommendations
            
            # Step 4: Create optimized content variations
            print("✨ Creating optimized content variations...")
            content_variations = self._create_content_variations(
                content, target_query, recommendations[:3]  # Top 3 recommendations
            )
            optimization_state['content_variations'] = content_variations
            
            # Step 5: Predict improvement potential
            print("📈 Calculating improvement potential...")
            improvement_prediction = self._predict_improvement_potential(
                current_probability, recommendations, gap_analysis
            )
            optimization_state['improvement_prediction'] = improvement_prediction
            
            # Step 6: Generate implementation roadmap
            print("🗺️ Creating implementation roadmap...")
            implementation_roadmap = self._create_implementation_roadmap(recommendations)
            optimization_state['implementation_roadmap'] = implementation_roadmap
            
            # Store optimization in history
            self.optimization_history.append(optimization_state)
            
            optimization_result = {
                'optimization_id': f"{self.agent_id}_{len(self.optimization_history)}",
                'target_query': target_query,
                'current_inclusion_probability': current_probability,
                'predicted_improvement': improvement_prediction,
                'priority_recommendations': recommendations[:5],  # Top 5 recommendations
                'quick_wins': [r for r in recommendations if r.get('implementation_difficulty') == 'easy'][:3],
                'content_variations': content_variations,
                'implementation_roadmap': implementation_roadmap,
                'optimization_summary': self._create_optimization_summary(optimization_state),
                'completion_timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Optimization complete!")
            print(f"📊 Potential improvement: +{improvement_prediction.get('total_improvement', 0):.1f}%")
            print(f"💡 Generated {len(recommendations)} recommendations")
            
            return optimization_result
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return {
                'error': str(e),
                'optimization_timestamp': datetime.now().isoformat(),
                'current_inclusion_probability': 0,
                'predicted_improvement': {'total_improvement': 0}
            }
    
    def _analyze_content_gaps(self, content: str, query: str, sub_intent: Dict, 
                             judge_evaluation: Dict) -> Dict:
        """Analyze gaps in current content compared to AI Overview requirements"""
        
        print("🔍 Analyzing content gaps...")
        
        gaps = {
            'structural_gaps': [],
            'content_gaps': [],
            'optimization_opportunities': [],
            'competitive_weaknesses': []
        }
        
        try:
            # Structural analysis
            if '<h1>' not in content and '<h2>' not in content:
                gaps['structural_gaps'].append({
                    'gap': 'missing_headings',
                    'description': 'Content lacks clear hierarchical structure',
                    'impact': 'medium',
                    'fix_difficulty': 'easy'
                })
            
            if '<ul>' not in content and '<ol>' not in content:
                gaps['structural_gaps'].append({
                    'gap': 'missing_lists',
                    'description': 'No bullet points or numbered lists for easy scanning',
                    'impact': 'medium',
                    'fix_difficulty': 'easy'
                })
            
            # Content completeness analysis
            content_length = len(content)
            if content_length < 300:
                gaps['content_gaps'].append({
                    'gap': 'insufficient_length',
                    'description': 'Content too brief for comprehensive coverage',
                    'impact': 'high',
                    'fix_difficulty': 'medium'
                })
            
            # Query alignment analysis
            query_words = query.lower().split()
            content_lower = content.lower()
            missing_keywords = [word for word in query_words if word not in content_lower]
            
            if missing_keywords:
                gaps['content_gaps'].append({
                    'gap': 'keyword_gaps',
                    'description': f'Missing key terms: {", ".join(missing_keywords)}',
                    'impact': 'high',
                    'fix_difficulty': 'easy'
                })
            
            # Sub-intent alignment
            if sub_intent and sub_intent.get('sub_intent'):
                sub_intent_text = sub_intent.get('sub_intent', '').lower()
                if sub_intent_text not in content_lower:
                    gaps['content_gaps'].append({
                        'gap': 'sub_intent_misalignment',
                        'description': f'Content doesn\'t address sub-intent: {sub_intent.get("sub_intent")}',
                        'impact': 'high',
                        'fix_difficulty': 'medium'
                    })
            
            # Judge evaluation insights
            judge_rank = judge_evaluation.get('candidate_rank')
            if judge_rank and judge_rank > 3:
                gaps['competitive_weaknesses'].append({
                    'gap': 'low_ranking',
                    'description': f'Content ranks #{judge_rank} against competitors',
                    'impact': 'high',
                    'fix_difficulty': 'hard'
                })
            
            # Quality score analysis
            rerank_score = judge_evaluation.get('candidate_score', 0)
            if rerank_score < 0.5:
                gaps['optimization_opportunities'].append({
                    'gap': 'low_quality_score',
                    'description': 'Content quality score below optimal threshold',
                    'impact': 'high',
                    'fix_difficulty': 'medium'
                })
            
            print(f"📊 Gap analysis complete:")
            print(f"   Structural gaps: {len(gaps['structural_gaps'])}")
            print(f"   Content gaps: {len(gaps['content_gaps'])}")
            print(f"   Optimization opportunities: {len(gaps['optimization_opportunities'])}")
            print(f"   Competitive weaknesses: {len(gaps['competitive_weaknesses'])}")
            
        except Exception as e:
            print(f"❌ Gap analysis failed: {e}")
            gaps['error'] = str(e)
        
        return gaps
    
    def _generate_optimization_recommendations(self, content: str, query: str, sub_intent: Dict,
                                             judge_evaluation: Dict, gap_analysis: Dict, 
                                             competitor_analysis: Dict) -> List[Dict]:
        """Generate specific, actionable optimization recommendations"""
        
        print("💡 Generating optimization recommendations...")
        
        recommendations = []
        
        try:
            current_probability = judge_evaluation.get('inclusion_probability', 0)
            
            # High-impact recommendations based on gaps
            for gap_category, gaps in gap_analysis.items():
                for gap in gaps:
                    if gap.get('impact') == 'high':
                        recommendation = self._create_recommendation_from_gap(gap, gap_category, query, sub_intent)
                        if recommendation:
                            recommendations.append(recommendation)
            
            # Content enhancement recommendations
            if current_probability < 30:
                recommendations.append({
                    'category': 'content_comprehensiveness',
                    'priority': 'critical',
                    'title': 'Expand content for comprehensive coverage',
                    'description': f'Current content too brief for AI Overview inclusion (probability: {current_probability:.1f}%)',
                    'specific_actions': [
                        f'Add detailed explanation of "{query}"',
                        'Include step-by-step instructions or examples',
                        'Address common related questions',
                        'Add relevant statistics or data points'
                    ],
                    'implementation_difficulty': 'medium',
                    'expected_improvement': '+20-35%',
                    'effort_required': 'high',
                    'implementation_time': '2-3 hours'
                })
            
            # Structure optimization recommendations
            if not any('<h' in content for content in [content]):  # Check for any heading tags
                recommendations.append({
                    'category': 'content_structure',
                    'priority': 'high',
                    'title': 'Add clear content hierarchy',
                    'description': 'Structure content with headings for better AI Overview extraction',
                    'specific_actions': [
                        'Add H2 headings for main sections',
                        'Use H3 subheadings for subsections',
                        'Create logical content flow',
                        'Add table of contents if appropriate'
                    ],
                    'implementation_difficulty': 'easy',
                    'expected_improvement': '+10-15%',
                    'effort_required': 'low',
                    'implementation_time': '30-45 minutes'
                })
            
            # Keyword optimization recommendations
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            missing_keywords = query_words - content_words
            
            if missing_keywords:
                recommendations.append({
                    'category': 'keyword_optimization',
                    'priority': 'high',
                    'title': 'Improve keyword alignment',
                    'description': f'Add missing query keywords: {", ".join(missing_keywords)}',
                    'specific_actions': [
                        f'Naturally incorporate: {", ".join(list(missing_keywords)[:3])}',
                        'Use keywords in headings and first paragraph',
                        'Add keyword variations and synonyms',
                        'Maintain natural language flow'
                    ],
                    'implementation_difficulty': 'easy',
                    'expected_improvement': '+8-12%',
                    'effort_required': 'low',
                    'implementation_time': '15-30 minutes'
                })
            
            # Competitive recommendations based on Judge evaluation
            top_competitors = judge_evaluation.get('top_competitors', [])
            if top_competitors:
                recommendations.append({
                    'category': 'competitive_optimization',
                    'priority': 'medium',
                    'title': 'Outperform top competitors',
                    'description': f'Analyze and exceed competitor content quality',
                    'specific_actions': [
                        f'Review top competitor: {top_competitors[0].get("title", "Unknown")}',
                        'Identify unique value propositions',
                        'Add content depth missing in competitors',
                        'Include more recent information or insights'
                    ],
                    'implementation_difficulty': 'medium',
                    'expected_improvement': '+5-15%',
                    'effort_required': 'medium',
                    'implementation_time': '1-2 hours'
                })
            
            # Sub-intent specific recommendations
            if sub_intent and sub_intent.get('user_motivation'):
                recommendations.append({
                    'category': 'user_intent_alignment',
                    'priority': 'high',
                    'title': 'Align with user motivation',
                    'description': f'Address specific user motivation: {sub_intent.get("user_motivation")}',
                    'specific_actions': [
                        f'Add section directly addressing: {sub_intent.get("user_motivation")}',
                        'Include practical examples or use cases',
                        'Answer potential follow-up questions',
                        'Add call-to-action relevant to user intent'
                    ],
                    'implementation_difficulty': 'medium',
                    'expected_improvement': '+12-18%',
                    'effort_required': 'medium',
                    'implementation_time': '45-90 minutes'
                })
            
            # Technical optimization recommendations
            if 'schema' not in content.lower() and 'json-ld' not in content.lower():
                recommendations.append({
                    'category': 'technical_seo',
                    'priority': 'low',
                    'title': 'Add structured data markup',
                    'description': 'Implement schema markup for better content understanding',
                    'specific_actions': [
                        'Add Article or FAQ schema markup',
                        'Include relevant JSON-LD structured data',
                        'Mark up key content sections',
                        'Validate schema implementation'
                    ],
                    'implementation_difficulty': 'hard',
                    'expected_improvement': '+3-8%',
                    'effort_required': 'high',
                    'implementation_time': '2-4 hours'
                })
            
            # Quality enhancement recommendations
            if judge_evaluation.get('candidate_score', 0) < 0.6:
                recommendations.append({
                    'category': 'content_quality',
                    'priority': 'high',
                    'title': 'Enhance content quality and clarity',
                    'description': 'Improve content quality score for better ranking',
                    'specific_actions': [
                        'Use clear, concise language',
                        'Add bullet points for key information',
                        'Include relevant examples and case studies',
                        'Improve readability and flow'
                    ],
                    'implementation_difficulty': 'medium',
                    'expected_improvement': '+10-20%',
                    'effort_required': 'medium',
                    'implementation_time': '1-2 hours'
                })
            
            # Sort recommendations by priority and expected improvement
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            recommendations.sort(key=lambda x: (
                priority_order.get(x.get('priority', 'medium'), 2),
                -float(x.get('expected_improvement', '+0%').replace('+', '').replace('%', '').split('-')[0])
            ))
            
            print(f"✅ Generated {len(recommendations)} optimization recommendations")
            for i, rec in enumerate(recommendations[:3]):
                print(f"   {i+1}. {rec.get('title', 'Unknown')} (Priority: {rec.get('priority', 'unknown')})")
            
        except Exception as e:
            print(f"❌ Recommendation generation failed: {e}")
            recommendations.append({
                'category': 'error',
                'priority': 'high',
                'title': 'Manual review required',
                'description': f'Automatic recommendation generation failed: {str(e)}',
                'specific_actions': ['Perform manual content analysis'],
                'implementation_difficulty': 'unknown',
                'expected_improvement': 'unknown'
            })
        
        return recommendations
    
    def _create_recommendation_from_gap(self, gap: Dict, gap_category: str, query: str, sub_intent: Dict) -> Optional[Dict]:
        """Create specific recommendation from identified gap"""
        
        gap_type = gap.get('gap', '')
        
        recommendation_templates = {
            'missing_headings': {
                'category': 'content_structure',
                'priority': 'high',
                'title': 'Add hierarchical content structure',
                'description': 'Implement clear heading structure for better content organization',
                'specific_actions': [
                    'Add H2 heading for main topic',
                    'Use H3 subheadings for subsections',
                    'Ensure logical content hierarchy'
                ],
                'implementation_difficulty': 'easy',
                'expected_improvement': '+8-12%'
            },
            'insufficient_length': {
                'category': 'content_completeness',
                'priority': 'critical',
                'title': 'Expand content comprehensiveness',
                'description': 'Increase content depth and coverage',
                'specific_actions': [
                    f'Add comprehensive explanation of "{query}"',
                    'Include relevant examples and use cases',
                    'Address related questions and topics'
                ],
                'implementation_difficulty': 'medium',
                'expected_improvement': '+25-40%'
            },
            'keyword_gaps': {
                'category': 'seo_optimization',
                'priority': 'high',
                'title': 'Improve keyword coverage',
                'description': gap.get('description', 'Add missing keywords'),
                'specific_actions': [
                    'Incorporate missing keywords naturally',
                    'Use keywords in headings and early paragraphs',
                    'Add semantic variations of key terms'
                ],
                'implementation_difficulty': 'easy',
                'expected_improvement': '+10-15%'
            }
        }
        
        if gap_type in recommendation_templates:
            template = recommendation_templates[gap_type].copy()
            template['effort_required'] = gap.get('fix_difficulty', 'medium')
            template['implementation_time'] = self._estimate_implementation_time(template['implementation_difficulty'])
            return template
        
        return None
    
    def _create_content_variations(self, original_content: str, query: str, 
                                  top_recommendations: List[Dict]) -> List[Dict]:
        """Create optimized content variations based on top recommendations"""
        
        print("✨ Creating content variations...")
        
        variations = []
        
        try:
            # Variation 1: Quick wins implementation
            quick_wins = [r for r in top_recommendations if r.get('implementation_difficulty') == 'easy']
            if quick_wins:
                variation1 = self._apply_quick_wins(original_content, query, quick_wins[:2])
                variations.append({
                    'variation_id': 'quick_wins',
                    'title': 'Quick Wins Optimization',
                    'description': 'Implements easy, high-impact improvements',
                    'content': variation1,
                    'applied_recommendations': [r.get('title', '') for r in quick_wins[:2]],
                    'estimated_improvement': '+15-25%',
                    'implementation_effort': 'low'
                })
            
            # Variation 2: Structure-focused optimization
            structure_recs = [r for r in top_recommendations if r.get('category') == 'content_structure']
            if structure_recs:
                variation2 = self._apply_structure_improvements(original_content, query)
                variations.append({
                    'variation_id': 'structure_optimized',
                    'title': 'Structure-Optimized Version',
                    'description': 'Enhanced content structure and organization',
                    'content': variation2,
                    'applied_recommendations': ['Hierarchical structure', 'Clear headings'],
                    'estimated_improvement': '+20-30%',
                    'implementation_effort': 'medium'
                })
            
            # Variation 3: Comprehensive optimization
            if len(top_recommendations) >= 3:
                variation3 = self._apply_comprehensive_optimization(original_content, query, top_recommendations[:3])
                variations.append({
                    'variation_id': 'comprehensive',
                    'title': 'Comprehensive Optimization',
                    'description': 'Implements multiple high-priority improvements',
                    'content': variation3,
                    'applied_recommendations': [r.get('title', '') for r in top_recommendations[:3]],
                    'estimated_improvement': '+35-50%',
                    'implementation_effort': 'high'
                })
            
            print(f"✅ Created {len(variations)} content variations")
            
        except Exception as e:
            print(f"❌ Content variation creation failed: {e}")
            variations.append({
                'variation_id': 'error',
                'title': 'Optimization Failed',
                'description': f'Content variation generation failed: {str(e)}',
                'content': original_content,
                'applied_recommendations': [],
                'estimated_improvement': '+0%'
            })
        
        return variations
    
    def _apply_quick_wins(self, content: str, query: str, quick_recommendations: List[Dict]) -> str:
        """Apply quick win optimizations to content"""
        
        optimized_content = content
        
        # Add basic structure if missing
        if '<h1>' not in optimized_content and '<h2>' not in optimized_content:
            # Add main heading
            optimized_content = f"# {query.title()}\n\n{optimized_content}"
        
        # Add bullet points for better readability
        if '<ul>' not in optimized_content and '<ol>' not in optimized_content:
            # Find a good place to add a list (after first paragraph)
            paragraphs = optimized_content.split('\n\n')
            if len(paragraphs) > 1:
                key_points = f"\n\n## Key Points\n\n- Important aspect 1\n- Important aspect 2\n- Important aspect 3\n\n"
                optimized_content = paragraphs[0] + key_points + '\n\n'.join(paragraphs[1:])
        
        # Ensure query keywords are present
        query_words = query.lower().split()
        content_lower = optimized_content.lower()
        for word in query_words:
            if word not in content_lower and len(word) > 2:  # Skip short words
                # Add keyword naturally in first paragraph
                lines = optimized_content.split('\n')
                if lines:
                    lines[0] = f"{lines[0]} This relates to {word} and relevant concepts."
                    optimized_content = '\n'.join(lines)
                break
        
        return optimized_content
    
    def _apply_structure_improvements(self, content: str, query: str) -> str:
        """Apply structure-focused improvements"""
        
        # Create well-structured version
        structured_content = f"""# {query.title()}

## Overview

{content.split('.')[0] if '.' in content else content[:200]}...

## Key Information

{content}

## Important Considerations

Consider these factors when dealing with {query}:

- Factor 1: Relevant consideration
- Factor 2: Important aspect  
- Factor 3: Key point to remember

## Summary

{query.capitalize()} involves multiple aspects that are important to understand for effective implementation.
"""
        
        return structured_content
    
    def _apply_comprehensive_optimization(self, content: str, query: str, recommendations: List[Dict]) -> str:
        """Apply comprehensive optimization based on recommendations"""
        
        # Start with structure improvements
        optimized = self._apply_structure_improvements(content, query)
        
        # Add FAQ section
        optimized += f"""

## Frequently Asked Questions

### What is {query}?
{query.capitalize()} refers to [detailed explanation based on original content].

### How does {query} work?
The process involves [step-by-step explanation].

### Why is {query} important?
{query.capitalize()} is important because [benefits and importance].

## Conclusion

Understanding {query} is essential for [relevant outcome]. This comprehensive guide covers the key aspects you need to know.
"""
        
        return optimized
    
    def _predict_improvement_potential(self, current_probability: float, recommendations: List[Dict], 
                                     gap_analysis: Dict) -> Dict:
        """Predict potential improvement from implementing recommendations"""
        
        prediction = {
            'current_probability': current_probability,
            'potential_improvements': [],
            'total_improvement': 0.0,
            'confidence_level': 'medium',
            'implementation_scenarios': {}
        }
        
        try:
            total_potential = 0.0
            
            # Calculate improvement from each recommendation
            for rec in recommendations:
                improvement_str = rec.get('expected_improvement', '+0%')
                # Extract numeric value (take average of range)
                import re
                numbers = re.findall(r'\d+', improvement_str)
                if numbers:
                    if len(numbers) > 1:
                        improvement = (int(numbers[0]) + int(numbers[1])) / 2
                    else:
                        improvement = int(numbers[0])
                    
                    total_potential += improvement
                    prediction['potential_improvements'].append({
                        'recommendation': rec.get('title', 'Unknown'),
                        'improvement': improvement,
                        'priority': rec.get('priority', 'medium')
                    })
            
            # Apply diminishing returns (can't exceed 100%)
            max_realistic_improvement = min(total_potential * 0.7, 100 - current_probability)
            prediction['total_improvement'] = max_realistic_improvement
            
            # Calculate scenarios
            prediction['implementation_scenarios'] = {
                'quick_wins_only': {
                    'improvement': min(max_realistic_improvement * 0.3, 25),
                    'effort': 'low',
                    'timeframe': '1-2 days'
                },
                'high_priority_recs': {
                    'improvement': min(max_realistic_improvement * 0.6, 40),
                    'effort': 'medium', 
                    'timeframe': '1-2 weeks'
                },
                'comprehensive_optimization': {
                    'improvement': max_realistic_improvement,
                    'effort': 'high',
                    'timeframe': '2-4 weeks'
                }
            }
            
            # Determine confidence level
            gap_count = sum(len(gaps) for gaps in gap_analysis.values() if isinstance(gaps, list))
            if gap_count <= 2:
                prediction['confidence_level'] = 'high'
            elif gap_count <= 5:
                prediction['confidence_level'] = 'medium'
            else:
                prediction['confidence_level'] = 'low'
            
            print(f"📈 Improvement prediction: +{max_realistic_improvement:.1f}% potential")
            
        except Exception as e:
            print(f"❌ Improvement prediction failed: {e}")
            prediction['error'] = str(e)
        
        return prediction
    
    def _create_implementation_roadmap(self, recommendations: List[Dict]) -> Dict:
        """Create step-by-step implementation roadmap"""
        
        roadmap = {
            'phases': [],
            'total_timeline': '2-4 weeks',
            'resource_requirements': {},
            'success_metrics': []
        }
        
        try:
            # Phase 1: Quick wins (Week 1)
            quick_wins = [r for r in recommendations if r.get('implementation_difficulty') == 'easy']
            if quick_wins:
                roadmap['phases'].append({
                    'phase': 1,
                    'title': 'Quick Wins Implementation',
                    'duration': '1-3 days',
                    'tasks': [r.get('title', 'Unknown task') for r in quick_wins[:3]],
                    'effort_required': 'low',
                    'expected_impact': '+15-25%'
                })
            
            # Phase 2: Medium priority improvements (Week 2-3)
            medium_tasks = [r for r in recommendations if r.get('implementation_difficulty') == 'medium']
            if medium_tasks:
                roadmap['phases'].append({
                    'phase': 2,
                    'title': 'Content Enhancement',
                    'duration': '1-2 weeks',
                    'tasks': [r.get('title', 'Unknown task') for r in medium_tasks[:3]],
                    'effort_required': 'medium',
                    'expected_impact': '+20-35%'
                })
            
            # Phase 3: Advanced optimizations (Week 3-4)
            advanced_tasks = [r for r in recommendations if r.get('implementation_difficulty') == 'hard']
            if advanced_tasks:
                roadmap['phases'].append({
                    'phase': 3,
                    'title': 'Advanced Optimization',
                    'duration': '1-2 weeks',
                    'tasks': [r.get('title', 'Unknown task') for r in advanced_tasks[:2]],
                    'effort_required': 'high',
                    'expected_impact': '+10-20%'
                })
            
            # Resource requirements
            roadmap['resource_requirements'] = {
                'content_writer': 'Required for phases 1-2',
                'seo_specialist': 'Required for phase 3',
                'developer': 'Required for technical implementations in phase 3',
                'estimated_total_hours': self._estimate_total_hours(recommendations)
            }
            
            # Success metrics
            roadmap['success_metrics'] = [
                'AI Overview inclusion probability increase',
                'Content ranking position improvement',
                'Search result click-through rate',
                'Content engagement metrics',
                'Judge Agent validation score'
            ]
            
        except Exception as e:
            print(f"❌ Roadmap creation failed: {e}")
            roadmap['error'] = str(e)
        
        return roadmap
    
    def _create_optimization_summary(self, optimization_state: Dict) -> Dict:
        """Create executive summary of optimization analysis"""
        
        try:
            judge_eval = optimization_state.get('judge_evaluation', {})
            recommendations = optimization_state.get('recommendations', [])
            improvement_pred = optimization_state.get('improvement_prediction', {})
            
            summary = {
                'current_status': {
                    'inclusion_probability': judge_eval.get('inclusion_probability', 0),
                    'content_rank': judge_eval.get('candidate_rank', 'Unknown'),
                    'content_length': len(optimization_state.get('original_content', '')),
                    'confidence_level': judge_eval.get('confidence_level', 'unknown')
                },
                'optimization_potential': {
                    'predicted_improvement': improvement_pred.get('total_improvement', 0),
                    'confidence': improvement_pred.get('confidence_level', 'medium'),
                    'priority_actions': len([r for r in recommendations if r.get('priority') in ['critical', 'high']])
                },
                'key_insights': self._extract_key_insights(optimization_state),
                'next_steps': [
                    'Implement quick wins for immediate improvement',
                    'Focus on high-priority recommendations',
                    'Monitor content performance after changes',
                    'Iterate based on results'
                ]
            }
            
            return summary
            
        except Exception as e:
            return {'error': f'Summary creation failed: {str(e)}'}
    
    def _extract_key_insights(self, optimization_state: Dict) -> List[str]:
        """Extract key insights from optimization analysis"""
        
        insights = []
        
        try:
            judge_eval = optimization_state.get('judge_evaluation', {})
            gap_analysis = optimization_state.get('gap_analysis', {})
            
            # Current performance insights
            probability = judge_eval.get('inclusion_probability', 0)
            if probability < 30:
                insights.append("Content currently has low AI Overview inclusion probability - significant optimization needed")
            elif probability < 60:
                insights.append("Content has moderate AI Overview potential - targeted improvements can boost performance")
            else:
                insights.append("Content already performing well - focus on fine-tuning for maximum impact")
            
            # Gap insights
            total_gaps = sum(len(gaps) for gaps in gap_analysis.values() if isinstance(gaps, list))
            if total_gaps > 5:
                insights.append("Multiple content gaps identified - comprehensive optimization recommended")
            elif total_gaps > 2:
                insights.append("Several optimization opportunities identified - prioritize high-impact changes")
            else:
                insights.append("Few gaps identified - content is relatively well-optimized")
            
            # Competitive insights
            top_competitors = judge_eval.get('top_competitors', [])
            if top_competitors:
                insights.append(f"Strong competition identified - need to outperform {len(top_competitors)} established competitors")
            
        except Exception as e:
            insights.append(f"Insight extraction failed: {str(e)}")
        
        return insights
    
    def _simplified_content_analysis(self, content: str, query: str) -> Dict:
        """Simplified content analysis when Judge Agent not available"""
        
        content_length = len(content)
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        keyword_overlap = len(query_words.intersection(content_words)) / len(query_words)
        
        # Simple scoring based on basic factors
        length_score = min(100, content_length / 5)  # 500 chars = 100 points
        keyword_score = keyword_overlap * 100
        structure_score = 50 if any(tag in content for tag in ['<h1>', '<h2>', '<h3>']) else 20
        
        overall_score = (length_score * 0.4 + keyword_score * 0.4 + structure_score * 0.2)
        
        return {
            'inclusion_probability': overall_score,
            'candidate_rank': 3 if overall_score > 60 else 5,
            'candidate_score': overall_score / 100,
            'confidence_level': 'medium',
            'top_competitors': [],
            'simplified_analysis': True
        }
    
    def _estimate_implementation_time(self, difficulty: str) -> str:
        """Estimate implementation time based on difficulty"""
        time_map = {
            'easy': '15-30 minutes',
            'medium': '1-2 hours', 
            'hard': '3-6 hours'
        }
        return time_map.get(difficulty, '1-2 hours')
    
    def _estimate_total_hours(self, recommendations: List[Dict]) -> str:
        """Estimate total hours for all recommendations"""
        total_mins = 0
        
        for rec in recommendations:
            difficulty = rec.get('implementation_difficulty', 'medium')
            if difficulty == 'easy':
                total_mins += 30
            elif difficulty == 'medium':
                total_mins += 90
            else:
                total_mins += 300
        
        hours = total_mins / 60
        return f"{hours:.1f} hours"

# ================================
# Optimizer Utility Functions
# ================================

def quick_content_optimization(content: str, query: str, azure_ai_client=None, 
                              judge_agent: JudgeAgentProduction = None) -> Dict:
    """Quick optimization function for single pieces of content"""
    
    try:
        optimizer = OptimizerAgent(azure_ai_client, judge_agent)
        return optimizer.optimize_content_for_aio(content, query)
    except Exception as e:
        print(f"❌ Quick optimization failed: {e}")
        return {
            'error': str(e),
            'optimization_timestamp': datetime.now().isoformat()
        }

# ================================
# Testing Function
# ================================

def test_optimizer_agent():
    """Test Optimizer Agent with sample content"""
    print("🧪 Testing Optimizer Agent...")
    
    test_content = """
    Digital banking allows customers to access banking services online. 
    It includes mobile apps and web portals for account management.
    """
    
    test_query = "digital banking services"
    
    try:
        result = quick_content_optimization(test_content, test_query)
        print("✅ Optimizer test completed!")
        print(f"📊 Recommendations generated: {len(result.get('priority_recommendations', []))}")
        print(f"🎯 Predicted improvement: {result.get('predicted_improvement', {}).get('total_improvement', 0):.1f}%")
        return result
    except Exception as e:
        print(f"❌ Optimizer test failed: {e}")
        return None

if __name__ == "__main__":
    # Run test
    test_optimizer_agent()
