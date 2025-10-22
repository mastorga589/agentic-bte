"""
Evidence-Weighted Scoring for BTE Results

This module provides comprehensive evidence scoring based on:
- Clinical trial phases and approval status
- Source reliability and multiplicity  
- Predicate relevance to query intent
- Study quality indicators

Replaces the simple 0.8 heuristic with nuanced scoring based on actual evidence.
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .predicate_strategy import QueryIntent

logger = logging.getLogger(__name__)


@dataclass 
class ScoringWeights:
    """Configurable weights for evidence scoring components"""
    clinical_evidence: float = 0.35      # Clinical trials, approval status
    predicate_relevance: float = 0.25    # How well predicate matches query intent  
    source_reliability: float = 0.20     # Provider quality and count
    result_multiplicity: float = 0.10    # Number of supporting edges/studies
    study_quality: float = 0.10          # Study size, completion status, etc.


class EvidenceScorer:
    """
    Computes evidence-weighted confidence scores for biomedical relationships
    based on BTE edge attributes, provenance, and query context
    """
    
    def __init__(self, weights: Optional[ScoringWeights] = None):
        """
        Initialize evidence scorer
        
        Args:
            weights: Custom scoring weights (uses defaults if None)
        """
        self.weights = weights or ScoringWeights()
        
        # Source reliability rankings (higher = more reliable)
        self.source_reliability_scores = {
            'infores:clinicaltrials': 0.9,
            'infores:aact': 0.85, 
            'infores:drugbank': 0.85,
            'infores:chebi': 0.8,
            'infores:umls': 0.75,
            'infores:mondo': 0.8,
            'infores:hgnc': 0.85,
            'infores:ensembl': 0.8,
            'infores:pubchem': 0.7,
            'infores:ctd': 0.75,
            'infores:biothings-explorer': 0.6,  # Aggregator
        }
    
    def score_result(self, result: Dict[str, Any], edges_data: Dict[str, Dict], 
                    predicate: str, query_intent: QueryIntent) -> float:
        """
        Calculate comprehensive evidence score for a single result
        
        Args:
            result: BTE result with node_bindings and analyses
            edges_data: Knowledge graph edges data (edge_id -> edge_info)  
            predicate: The predicate used in this result
            query_intent: Detected query intent
            
        Returns:
            Evidence score (0.0-1.0)
        """
        try:
            # Extract edge IDs that support this result
            edge_ids = self._extract_supporting_edge_ids(result)
            if not edge_ids:
                return self._fallback_score(predicate, query_intent)
            
            # Collect edge attributes from all supporting edges
            all_edge_attributes = []
            all_sources = []
            
            for edge_id in edge_ids:
                if edge_id in edges_data:
                    edge_info = edges_data[edge_id]
                    attributes = edge_info.get('attributes', [])
                    sources = edge_info.get('sources', [])
                    
                    all_edge_attributes.extend(attributes)
                    all_sources.extend(sources)
            
            if not all_edge_attributes and not all_sources:
                return self._fallback_score(predicate, query_intent)
            
            # Calculate component scores
            clinical_score = self._calculate_clinical_evidence_score(all_edge_attributes)
            predicate_score = self._calculate_predicate_relevance_score(predicate, query_intent) 
            source_score = self._calculate_source_reliability_score(all_sources)
            multiplicity_score = self._calculate_multiplicity_score(edge_ids, all_edge_attributes)
            study_quality_score = self._calculate_study_quality_score(all_edge_attributes)
            
            # Weighted composite score
            composite_score = (
                self.weights.clinical_evidence * clinical_score +
                self.weights.predicate_relevance * predicate_score +
                self.weights.source_reliability * source_score +
                self.weights.result_multiplicity * multiplicity_score +
                self.weights.study_quality * study_quality_score
            )
            
            # Clamp to [0, 1] range
            final_score = max(0.0, min(1.0, composite_score))
            
            logger.debug(f"Evidence score breakdown - Clinical: {clinical_score:.3f}, "
                        f"Predicate: {predicate_score:.3f}, Source: {source_score:.3f}, "
                        f"Multiplicity: {multiplicity_score:.3f}, Quality: {study_quality_score:.3f}, "
                        f"Final: {final_score:.3f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating evidence score: {str(e)}")
            return self._fallback_score(predicate, query_intent)
    
    def _extract_supporting_edge_ids(self, result: Dict[str, Any]) -> List[str]:
        """Extract edge IDs that support this result"""
        edge_ids = []
        
        analyses = result.get('analyses', [])
        for analysis in analyses:
            edge_bindings = analysis.get('edge_bindings', {})
            for edge_key, edge_list in edge_bindings.items():
                for edge_binding in edge_list:
                    edge_id = edge_binding.get('id')
                    if edge_id:
                        edge_ids.append(edge_id)
        
        return edge_ids
    
    def _calculate_clinical_evidence_score(self, attributes: List[Dict]) -> float:
        """Calculate score based on clinical evidence strength"""
        score = 0.0
        
        for attr in attributes:
            attr_type = attr.get('attribute_type_id', '')
            value = attr.get('value')
            
            # Clinical trial phase (highest weight)
            if attr_type == 'biolink:max_research_phase' and isinstance(value, (int, float)):
                if value >= 4:
                    score += 0.5  # FDA approved
                elif value >= 3:
                    score += 0.4  # Phase 3
                elif value >= 2:
                    score += 0.3  # Phase 2
                elif value >= 1:
                    score += 0.2  # Phase 1
            
            # Approval status
            elif attr_type == 'biolink:clinical_approval_status':
                if 'approved_for_condition' in str(value).lower():
                    score += 0.4
                elif 'indicated' in str(value).lower():
                    score += 0.3
            
            # Clinical trial evidence
            elif attr_type == 'clinical_trial_tested_intervention':
                if str(value).lower() == 'yes':
                    score += 0.2
            
            # Supporting studies
            elif attr_type == 'biolink:supporting_study':
                score += 0.1  # Any supporting study adds value
                
                # Check nested study attributes
                study_attrs = attr.get('attributes', [])
                for study_attr in study_attrs:
                    study_attr_type = study_attr.get('attribute_type_id', '')
                    study_value = study_attr.get('value')
                    
                    # Clinical trial phase within study
                    if study_attr_type == 'clinical_trial_phase' and isinstance(study_value, (int, float)):
                        if study_value >= 4:
                            score += 0.2
                        elif study_value >= 3:
                            score += 0.15
                        elif study_value >= 2:
                            score += 0.1
                    
                    # Study completion status
                    elif study_attr_type == 'clinical_trial_status':
                        if str(study_value).upper() == 'COMPLETED':
                            score += 0.1
            
            # Knowledge level (assertion vs prediction)
            elif attr_type == 'biolink:knowledge_level':
                if 'assertion' in str(value).lower():
                    score += 0.15
                elif 'prediction' in str(value).lower():
                    score += 0.05
        
        return min(score, 1.0)
    
    def _calculate_predicate_relevance_score(self, predicate: str, query_intent: QueryIntent) -> float:
        """Calculate score based on how well predicate matches query intent"""
        # Import predicate selector for relevance weights
        from .predicate_strategy import PredicateSelector
        
        temp_selector = PredicateSelector()
        return temp_selector.get_predicate_relevance_score(predicate, query_intent)
    
    def _calculate_source_reliability_score(self, sources: List[Dict]) -> float:
        """Calculate score based on data source reliability"""
        if not sources:
            return 0.3  # Default modest score for unknown sources
        
        max_reliability = 0.0
        primary_sources = 0
        supporting_sources = 0
        
        for source in sources:
            resource_id = source.get('resource_id', '')
            resource_role = source.get('resource_role', '')
            
            # Get base reliability for this source
            base_reliability = self.source_reliability_scores.get(resource_id, 0.5)
            
            # Boost based on role
            if resource_role == 'primary_knowledge_source':
                adjusted_reliability = base_reliability + 0.1
                primary_sources += 1
            elif resource_role == 'supporting_data_source':
                adjusted_reliability = base_reliability
                supporting_sources += 1
            elif resource_role == 'aggregator_knowledge_source':
                adjusted_reliability = base_reliability - 0.1  # Slightly lower for aggregators
            else:
                adjusted_reliability = base_reliability
            
            max_reliability = max(max_reliability, adjusted_reliability)
        
        # Bonus for multiple supporting sources
        if supporting_sources > 1:
            max_reliability += min(0.2, 0.05 * supporting_sources)
        
        # Bonus for having primary sources
        if primary_sources > 0:
            max_reliability += 0.1
        
        return min(max_reliability, 1.0)
    
    def _calculate_multiplicity_score(self, edge_ids: List[str], attributes: List[Dict]) -> float:
        """Calculate score based on number of supporting evidence pieces"""
        score = 0.0
        
        # Multiple supporting edges
        if len(edge_ids) > 1:
            score += min(0.3, 0.05 * len(edge_ids))
        
        # Multiple supporting studies
        study_count = sum(1 for attr in attributes if attr.get('attribute_type_id') == 'biolink:supporting_study')
        if study_count > 0:
            score += min(0.2, 0.1 * study_count)
        
        # Multiple clinical trials
        trial_count = sum(1 for attr in attributes if 'clinical_trial' in attr.get('attribute_type_id', '').lower())
        if trial_count > 0:
            score += min(0.2, 0.05 * trial_count)
        
        return min(score, 1.0)
    
    def _calculate_study_quality_score(self, attributes: List[Dict]) -> float:
        """Calculate score based on study quality indicators"""
        score = 0.0
        
        for attr in attributes:
            attr_type = attr.get('attribute_type_id', '')
            value = attr.get('value')
            
            # Check for study quality indicators in nested attributes
            if attr_type == 'biolink:supporting_study':
                study_attrs = attr.get('attributes', [])
                for study_attr in study_attrs:
                    study_attr_type = study_attr.get('attribute_type_id', '')
                    study_value = study_attr.get('value')
                    
                    # Study size (diminishing returns)
                    if study_attr_type == 'study_size' and isinstance(study_value, (int, float)):
                        if study_value > 0:
                            size_score = min(0.3, math.log10(study_value) / 4.0)  # Log scale
                            score += size_score
                    
                    # Study completion
                    elif study_attr_type == 'clinical_trial_status':
                        if str(study_value).upper() == 'COMPLETED':
                            score += 0.2
                    
                    # Primary purpose is treatment (for therapeutic queries)
                    elif study_attr_type == 'primary_purpose':
                        if str(study_value).upper() == 'TREATMENT':
                            score += 0.15
                    
                    # Age range coverage (adults preferred over pediatric only)
                    elif study_attr_type == 'adult' and study_value is True:
                        score += 0.1
            
            # Penalize adverse signals
            elif 'warning' in attr_type.lower() or 'adverse' in attr_type.lower():
                score -= 0.1  # Small penalty for safety concerns
        
        return max(0.0, min(score, 1.0))
    
    def _fallback_score(self, predicate: str, query_intent: QueryIntent) -> float:
        """Fallback scoring when no edge attributes are available"""
        # Base score on predicate relevance when no other evidence exists
        predicate_score = self._calculate_predicate_relevance_score(predicate, query_intent)
        
        # Conservative scoring - assume modest evidence quality
        if predicate_score >= 0.8:  # High relevance predicates
            return 0.6
        elif predicate_score >= 0.6:  # Medium relevance predicates  
            return 0.5
        elif predicate_score >= 0.4:  # Lower relevance predicates
            return 0.4
        else:  # Generic predicates like related_to
            return 0.3
    
    def get_score_explanation(self, result: Dict[str, Any], edges_data: Dict[str, Dict],
                            predicate: str, query_intent: QueryIntent) -> Dict[str, Any]:
        """
        Get detailed breakdown of evidence scoring for transparency
        
        Args:
            result: BTE result
            edges_data: Knowledge graph edges data
            predicate: The predicate used
            query_intent: Query intent
            
        Returns:
            Dictionary with score breakdown and explanations
        """
        try:
            edge_ids = self._extract_supporting_edge_ids(result)
            
            if not edge_ids:
                return {
                    'total_score': self._fallback_score(predicate, query_intent),
                    'explanation': 'Fallback scoring - no detailed evidence available',
                    'components': {}
                }
            
            # Collect attributes and sources
            all_edge_attributes = []
            all_sources = []
            
            for edge_id in edge_ids:
                if edge_id in edges_data:
                    edge_info = edges_data[edge_id]
                    all_edge_attributes.extend(edge_info.get('attributes', []))
                    all_sources.extend(edge_info.get('sources', []))
            
            # Calculate component scores with explanations
            clinical_score = self._calculate_clinical_evidence_score(all_edge_attributes)
            predicate_score = self._calculate_predicate_relevance_score(predicate, query_intent)
            source_score = self._calculate_source_reliability_score(all_sources)
            multiplicity_score = self._calculate_multiplicity_score(edge_ids, all_edge_attributes)
            study_quality_score = self._calculate_study_quality_score(all_edge_attributes)
            
            total_score = (
                self.weights.clinical_evidence * clinical_score +
                self.weights.predicate_relevance * predicate_score +
                self.weights.source_reliability * source_score +
                self.weights.result_multiplicity * multiplicity_score +
                self.weights.study_quality * study_quality_score
            )
            
            return {
                'total_score': max(0.0, min(1.0, total_score)),
                'components': {
                    'clinical_evidence': clinical_score,
                    'predicate_relevance': predicate_score,
                    'source_reliability': source_score,
                    'result_multiplicity': multiplicity_score,
                    'study_quality': study_quality_score
                },
                'weights': {
                    'clinical_evidence': self.weights.clinical_evidence,
                    'predicate_relevance': self.weights.predicate_relevance,
                    'source_reliability': self.weights.source_reliability,
                    'result_multiplicity': self.weights.result_multiplicity,
                    'study_quality': self.weights.study_quality
                },
                'evidence_count': {
                    'supporting_edges': len(edge_ids),
                    'attributes': len(all_edge_attributes),
                    'sources': len(all_sources)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating score explanation: {str(e)}")
            return {
                'total_score': self._fallback_score(predicate, query_intent),
                'explanation': f'Error in scoring: {str(e)}',
                'components': {}
            }


def create_evidence_scorer(meta_kg: Optional[Dict[str, Any]] = None, 
                          weights: Optional[ScoringWeights] = None) -> EvidenceScorer:
    """
    Factory function to create an evidence scorer
    
    Args:
        meta_kg: Meta knowledge graph (optional, for predicate support analysis)
        weights: Custom scoring weights
        
    Returns:
        Configured EvidenceScorer instance
    """
    # The meta_kg parameter is accepted but not currently used by EvidenceScorer
    # This prevents warnings while maintaining compatibility
    return EvidenceScorer(weights=weights)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    scorer = EvidenceScorer()
    
    # Mock result with clinical trial attributes
    mock_result = {
        'analyses': [{
            'edge_bindings': {
                'e0': [{'id': 'edge_123'}]
            }
        }]
    }
    
    mock_edges = {
        'edge_123': {
            'attributes': [
                {
                    'attribute_type_id': 'biolink:max_research_phase',
                    'value': 4
                },
                {
                    'attribute_type_id': 'biolink:clinical_approval_status', 
                    'value': 'biolink:approved_for_condition'
                }
            ],
            'sources': [
                {
                    'resource_id': 'infores:clinicaltrials',
                    'resource_role': 'primary_knowledge_source'
                }
            ]
        }
    }
    
    score = scorer.score_result(mock_result, mock_edges, 'biolink:treats', QueryIntent.THERAPEUTIC)
    print(f"Mock therapeutic result score: {score:.3f}")
    
    explanation = scorer.get_score_explanation(mock_result, mock_edges, 'biolink:treats', QueryIntent.THERAPEUTIC)
    print(f"Score breakdown: {explanation}")