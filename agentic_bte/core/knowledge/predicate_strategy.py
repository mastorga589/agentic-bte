"""
Predicate Selection Strategy for Biomedical Queries

This module provides intelligent predicate selection based on query intent,
Biolink model hierarchy, and BTE meta-knowledge graph provider support.

Uses a 3-tier approach:
- Primary: High-specificity predicates with strong semantic meaning
- Secondary: Medium-specificity predicates for broader context  
- Fallback: biolink:related_to as universal fallback for any missed connections
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ...config.settings import get_settings

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of biomedical query intents"""
    THERAPEUTIC = "therapeutic"  # Drug treatment queries
    GENETIC = "genetic"         # Gene-disease associations
    MECHANISM = "mechanism"     # Drug/protein interactions, pathways
    GENERAL = "general"         # Mixed or unclear intent


@dataclass
class PredicateConfig:
    """Configuration for predicate selection"""
    max_predicates_per_subquery: int = 4
    min_results_threshold: int = 10  # Add secondary tier if primary < this
    fallback_threshold: int = 5      # Add fallback if secondary < this
    enable_meta_kg_filtering: bool = True
    min_provider_support: int = 1    # Minimum edges in meta-KG to use predicate


class PredicateSelector:
    """
    Intelligent predicate selection based on query intent and meta-KG support
    """
    
    # Predicate tiers by query intent - tuned for specificity and performance
    PREDICATE_TIERS = {
        QueryIntent.THERAPEUTIC: {
            'primary': [
                'biolink:treats',
                'biolink:applied_to_treat',
                'biolink:treated_by'
            ],
            'secondary': [
                'biolink:in_clinical_trials_for',
                'biolink:treats_or_applied_or_studied_to_treat',
                'biolink:affects'
            ]
        },
        QueryIntent.GENETIC: {
            'primary': [
                'biolink:gene_associated_with_condition',
                'biolink:condition_associated_with_gene',
                'biolink:genetically_associated_with'
            ],
            'secondary': [
                'biolink:causes',
                'biolink:contributes_to',
                'biolink:affects'
            ]
        },
        QueryIntent.MECHANISM: {
            'primary': [
                'biolink:interacts_with',
                'biolink:directly_physically_interacts_with',
                'biolink:regulates'
            ],
            'secondary': [
                'biolink:physically_interacts_with',
                'biolink:modulates',
                'biolink:affects'
            ]
        },
        QueryIntent.GENERAL: {
            'primary': [
                'biolink:affects',
                'biolink:interacts_with'
            ],
            'secondary': [
                'biolink:correlated_with',
                'biolink:coexists_with'
            ]
        }
    }
    
    # Universal fallback predicate (only used if allowed and supported)
    FALLBACK_PREDICATE = 'biolink:related_to'
    
    def __init__(self, meta_kg: Optional[Dict] = None, config: Optional[PredicateConfig] = None):
        """
        Initialize predicate selector
        
        Args:
            meta_kg: BTE meta knowledge graph data
            config: Predicate selection configuration
        """
        self.meta_kg = meta_kg or {}
        self.config = config or PredicateConfig()
        self._predicate_support_cache = {}
        
        # Build predicate support mapping from meta-KG
        self._build_predicate_support_map()
    
    def _build_predicate_support_map(self):
        """Build mapping of (subject_cat, predicate, object_cat) -> provider_count"""
        if not self.meta_kg:
            logger.debug("No meta-KG provided, skipping predicate support analysis")
            return
            
        edges = self.meta_kg.get('edges', [])
        logger.info(f"Building predicate support map from {len(edges)} meta-KG edges")
        
        for edge in edges:
            subject_cat = edge.get('subject')
            object_cat = edge.get('object')
            predicate = edge.get('predicate')
            
            if subject_cat and object_cat and predicate:
                key = (subject_cat, predicate, object_cat)
                self._predicate_support_cache[key] = self._predicate_support_cache.get(key, 0) + 1
        
        logger.info(f"Built support map with {len(self._predicate_support_cache)} predicate patterns")
    
    def get_predicate_support(self, subject_category: str, predicate: str, object_category: str) -> int:
        """Get number of providers supporting this predicate pattern"""
        key = (subject_category, predicate, object_category)
        return self._predicate_support_cache.get(key, 0)
    
    def detect_query_intent(self, query: str, entities: List[Dict]) -> QueryIntent:
        """
        Detect the primary intent of a biomedical query
        
        Args:
            query: Query text
            entities: Extracted entities with types
            
        Returns:
            Detected query intent
        """
        query_lower = query.lower()
        
        # Therapeutic intent keywords
        if any(keyword in query_lower for keyword in [
            'treat', 'drug', 'medicine', 'therapeutic', 'therapy', 'medication',
            'compound', 'chemical', 'cure', 'heal', 'remedy'
        ]):
            return QueryIntent.THERAPEUTIC
        
        # Genetic intent keywords  
        if any(keyword in query_lower for keyword in [
            'gene', 'genetic', 'hereditary', 'inherited', 'mutation',
            'allele', 'chromosome', 'dna', 'genome'
        ]):
            return QueryIntent.GENETIC
        
        # Mechanism intent keywords
        if any(keyword in query_lower for keyword in [
            'mechanism', 'pathway', 'process', 'interact', 'affect', 'influence',
            'regulate', 'modulate', 'target', 'bind', 'inhibit', 'activate'
        ]):
            return QueryIntent.MECHANISM
        
        # Check entity types for additional clues
        entity_types = [entity.get('type', '').lower() for entity in entities]
        
        if any('drug' in t or 'chemical' in t for t in entity_types):
            if any('disease' in t or 'condition' in t for t in entity_types):
                return QueryIntent.THERAPEUTIC
            elif any('protein' in t or 'process' in t for t in entity_types):
                return QueryIntent.MECHANISM
        
        if any('gene' in t for t in entity_types):
            return QueryIntent.GENETIC
        
        return QueryIntent.GENERAL
    
    def select_predicates(self, query_intent: QueryIntent, subject_category: str, 
                         object_category: str, expected_results: int = 0) -> List[Tuple[str, float]]:
        """
        Select optimal predicates for a query based on intent and meta-KG support
        
        Args:
            query_intent: Detected intent of the query
            subject_category: TRAPI subject category (e.g., 'biolink:Disease')
            object_category: TRAPI object category (e.g., 'biolink:SmallMolecule')
            expected_results: Expected number of results (for adaptive selection)
            
        Returns:
            List of (predicate, priority_score) tuples, ordered by priority
        """
        logger.info(f"Selecting predicates for {query_intent.value} query: {subject_category} -> {object_category}")
        
        selected_predicates = []
        
        # Get predicate tiers for this intent
        tiers = self.PREDICATE_TIERS.get(query_intent, self.PREDICATE_TIERS[QueryIntent.GENERAL])
        
        # Primary tier - always include if supported
        primary_predicates = self._filter_supported_predicates(
            tiers['primary'], subject_category, object_category
        )
        for pred in primary_predicates[:2]:  # Limit to top 2 primary
            priority = 1.0 - (primary_predicates.index(pred) * 0.1)  # 1.0, 0.9, etc.
            selected_predicates.append((pred, priority))
        
        # Secondary tier - add if we need broader coverage
        if len(selected_predicates) < 2 or expected_results < self.config.min_results_threshold:
            secondary_predicates = self._filter_supported_predicates(
                tiers['secondary'], subject_category, object_category
            )
            for pred in secondary_predicates[:2]:  # Limit to top 2 secondary
                priority = 0.7 - (secondary_predicates.index(pred) * 0.1)  # 0.7, 0.6, etc.
                selected_predicates.append((pred, priority))
        
# Fallback predicate - add only if not excluded and supported
        try:
            settings = get_settings()
            excluded: set[str] = set(settings.excluded_predicates or [])
        except Exception:
            excluded = set()
        if len(selected_predicates) < self.config.max_predicates_per_subquery and self.FALLBACK_PREDICATE not in excluded:
            fallback_support = self.get_predicate_support(subject_category, self.FALLBACK_PREDICATE, object_category)
            if fallback_support > 0:
                selected_predicates.append((self.FALLBACK_PREDICATE, 0.3))
        
        # Sort by priority, drop duplicates, and limit total count
        selected_predicates.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        deduped: List[Tuple[str, float]] = []
        for pred, pr in selected_predicates:
            if pred in excluded or pred in seen:
                continue
            seen.add(pred)
            deduped.append((pred, pr))
        selected_predicates = deduped[:self.config.max_predicates_per_subquery]
        
        logger.info(f"Selected {len(selected_predicates)} predicates: {[p[0] for p in selected_predicates]}")
        return selected_predicates
    
    def _filter_supported_predicates(self, predicates: List[str], subject_category: str, 
                                   object_category: str) -> List[str]:
        """Filter predicates based on meta-KG provider support and user settings"""
        # Load excluded predicates from settings
        try:
            settings = get_settings()
            excluded: set[str] = set(settings.excluded_predicates or [])
        except Exception:
            excluded = set()
        
        # Respect meta-KG filtering if enabled
        filtered: List[str] = []
        for predicate in predicates:
            if predicate in excluded:
                logger.debug(f"{predicate}: excluded by settings")
                continue
            if self.config.enable_meta_kg_filtering:
                support_count = self.get_predicate_support(subject_category, predicate, object_category)
                if support_count < self.config.min_provider_support:
                    logger.debug(f"{predicate}: {support_count} providers (filtered out)")
                    continue
                logger.debug(f"{predicate}: {support_count} providers")
            filtered.append(predicate)
        
        return filtered
    
    def get_predicate_relevance_score(self, predicate: str, query_intent: QueryIntent) -> float:
        """
        Get relevance score for a predicate given the query intent
        
        Args:
            predicate: Biolink predicate
            query_intent: Query intent type
            
        Returns:
            Relevance score (0.0-1.0)
        """
        # Predicate relevance weights by intent
        relevance_weights = {
            QueryIntent.THERAPEUTIC: {
                'biolink:treats': 1.0,
                'biolink:treated_by': 1.0,
                'biolink:applied_to_treat': 0.9,
                'biolink:treats_or_applied_or_studied_to_treat': 0.8,
                'biolink:in_clinical_trials_for': 0.7,
                'biolink:tested_by_clinical_trials_of': 0.7,
                'biolink:studied_to_treat': 0.6,
                'biolink:affects': 0.4,
                'biolink:related_to': 0.2
            },
            QueryIntent.GENETIC: {
                'biolink:gene_associated_with_condition': 1.0,
                'biolink:condition_associated_with_gene': 1.0,
                'biolink:predisposes_to_condition': 0.9,
                'biolink:causes': 0.8,
                'biolink:contributes_to': 0.7,
                'biolink:genetically_associated_with': 0.6,
                'biolink:associated_with_increased_likelihood_of': 0.5,
                'biolink:correlated_with': 0.4,
                'biolink:positively_correlated_with': 0.4,
                'biolink:related_to': 0.2
            },
            QueryIntent.MECHANISM: {
                'biolink:directly_physically_interacts_with': 1.0,
                'biolink:affects': 0.9,
                'biolink:regulates': 0.8,
                'biolink:physically_interacts_with': 0.7,
                'biolink:interacts_with': 0.6,
                'biolink:disrupts': 0.5,
                'biolink:modulates': 0.5,
                'biolink:influences': 0.4,
                'biolink:correlated_with': 0.3,
                'biolink:related_to': 0.2
            },
            QueryIntent.GENERAL: {
                'biolink:affects': 0.8,
                'biolink:associated_with': 0.7,
                'biolink:correlated_with': 0.6,
                'biolink:interacts_with': 0.5,
                'biolink:coexists_with': 0.4,
                'biolink:related_to': 0.3
            }
        }
        
        return relevance_weights.get(query_intent, {}).get(predicate, 0.3)


def create_predicate_selector(meta_kg: Optional[Dict] = None) -> PredicateSelector:
    """
    Factory function to create a predicate selector with meta-KG support
    
    Args:
        meta_kg: BTE meta knowledge graph data
        
    Returns:
        Configured PredicateSelector instance
    """
    return PredicateSelector(meta_kg=meta_kg)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    selector = PredicateSelector()
    
    # Test therapeutic query
    therapeutic_predicates = selector.select_predicates(
        QueryIntent.THERAPEUTIC,
        "biolink:Disease", 
        "biolink:SmallMolecule"
    )
    print(f"Therapeutic predicates: {therapeutic_predicates}")
    
    # Test genetic query
    genetic_predicates = selector.select_predicates(
        QueryIntent.GENETIC,
        "biolink:Gene",
        "biolink:Disease"
    )
    print(f"Genetic predicates: {genetic_predicates}")