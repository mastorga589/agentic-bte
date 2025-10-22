#!/usr/bin/env python3
"""
Entity Filtering Enhancement for BioNER System

This module provides intelligent filtering to remove generic, non-specific 
biomedical terms from entity recognition results, preventing them from 
being used in TRAPI queries.

Key improvements:
1. Filter out generic action words like "targeting", "treating", "affecting"
2. Remove overly broad terms like "drugs", "genes", "diseases" without context
3. Preserve specific biomedical entities like "Brucellosis", "translation"
4. Maintain backward compatibility with existing BioNER system
"""

import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)

class BiomedicaleEntityFilter:
    """
    Intelligent filter for removing generic biomedical terms from entity recognition
    """
    
    # Generic terms that should be filtered out
    GENERIC_TERMS = {
        # Action words
        'targeting', 'treating', 'affecting', 'modulating', 'inhibiting', 'activating',
        'regulating', 'binding', 'interacting', 'expressing', 'producing',
        'treat', 'treating', 'target', 'targeting', 'affect', 'affecting',
        
        # Overly broad categories
        'drugs', 'medications', 'treatments', 'therapies', 'compounds', 'substances',
        'genes', 'proteins', 'enzymes', 'receptors', 'factors',
        'diseases', 'disorders', 'conditions', 'syndromes', 'symptoms',
        'processes', 'pathways', 'mechanisms', 'functions', 'activities',
        'cells', 'tissues', 'organs', 'systems',
        
        # Generic biological concepts that need specificity
        'genetic variations', 'genetic factors', 'genetic polymorphisms',
        'protein interactions', 'molecular mechanisms', 'biological processes',
        'cellular processes', 'metabolic processes', 'signaling pathways',
        'gene expression', 'protein expression', 'enzyme activity',
        'receptor binding', 'drug interactions', 'therapeutic effects',
        
        # Non-specific terms
        'by', 'with', 'through', 'via', 'using', 'from', 'to', 'for', 'in', 'on', 'at',
        'that', 'which', 'what', 'how', 'when', 'where', 'why',
        'can', 'could', 'may', 'might', 'will', 'would', 'should',
        'do', 'does', 'did', 'have', 'has', 'had', 'are', 'is', 'was', 'were',
        
        # Common English words that might be extracted
        'the', 'and', 'or', 'but', 'if', 'then', 'also', 'other', 'another',
        'more', 'most', 'some', 'any', 'all', 'each', 'every', 'both', 'either',
    }
    
    # Specific patterns that indicate generic terms
    GENERIC_PATTERNS = {
        # Plural forms without context
        r'^.*s$',  # Words ending in 's' that are too generic
    }
    
    # Terms that should be preserved even if they match generic patterns
    PRESERVE_TERMS = {
        # Specific diseases
        'brucellosis', 'diabetes', 'cancer', 'alzheimer', 'parkinson', 'huntington',
        'tuberculosis', 'malaria', 'pneumonia', 'asthma', 'arthritis',
        
        # Specific biological processes
        'translation', 'transcription', 'replication', 'apoptosis', 'mitosis',
        'glycolysis', 'photosynthesis', 'respiration', 'metabolism',
        
        # Specific drug names
        'aspirin', 'metformin', 'insulin', 'penicillin', 'morphine',
        'acetaminophen', 'ibuprofen', 'warfarin',
        
        # Specific gene/protein names (often short)
        'tp53', 'brca1', 'brca2', 'apoe', 'cftr', 'egfr', 'vegf',
        
        # Chemical compounds
        'glucose', 'cholesterol', 'dopamine', 'serotonin', 'acetylcholine',
    }
    
    def __init__(self):
        """Initialize the entity filter"""
        self.generic_terms_lower = {term.lower() for term in self.GENERIC_TERMS}
        self.preserve_terms_lower = {term.lower() for term in self.PRESERVE_TERMS}
    
    def filter_entities(self, entities: List[str]) -> List[str]:
        """
        Filter out generic terms from entity list
        
        Args:
            entities: List of entity text strings
            
        Returns:
            Filtered list of specific biomedical entities
        """
        filtered = []
        
        for entity in entities:
            if self.should_preserve_entity(entity):
                filtered.append(entity)
            else:
                logger.debug(f"Filtered out generic term: '{entity}'")
        
        logger.info(f"Filtered entities: {len(entities)} -> {len(filtered)} (removed {len(entities) - len(filtered)} generic terms)")
        return filtered
    
    def should_preserve_entity(self, entity: str) -> bool:
        """
        Determine if an entity should be preserved
        
        Args:
            entity: Entity text to evaluate
            
        Returns:
            True if entity should be preserved, False if it should be filtered
        """
        entity_clean = entity.strip().lower()
        
        # Always preserve terms in the preserve list
        if entity_clean in self.preserve_terms_lower:
            return True
        
        # Filter out generic terms
        if entity_clean in self.generic_terms_lower:
            return False
        
        # Additional heuristics
        
        # Filter out very short terms that are likely generic
        if len(entity_clean) <= 2:
            return False
        
        # Filter out terms that are just numbers or contain only numbers and basic punctuation
        if entity_clean.replace(' ', '').replace('-', '').replace('.', '').isdigit():
            return False
        
        # Preserve terms that look like specific biomedical identifiers
        # (contain capital letters, numbers, or specific patterns)
        original_entity = entity.strip()
        
        # Gene symbols (often uppercase, short)
        if len(original_entity) <= 8 and any(c.isupper() for c in original_entity):
            return True
        
        # Chemical formulas or compound names (contain numbers or specific patterns)
        if any(char.isdigit() for char in original_entity):
            return True
        
        # Multi-word terms that are likely specific
        if ' ' in entity_clean and len(entity_clean.split()) >= 2:
            # Check if all words are generic
            words = entity_clean.split()
            specific_words = [w for w in words if w not in self.generic_terms_lower]
            
            # For multi-word terms, require at least one specific word AND
            # the combination must not be a generic phrase
            if len(specific_words) >= 1 and entity_clean not in self.generic_terms_lower:
                # Additional check: avoid generic multi-word phrases
                generic_phrases = {
                    'genetic variations', 'genetic factors', 'protein interactions',
                    'molecular mechanisms', 'biological processes', 'cellular processes',
                    'metabolic processes', 'signaling pathways', 'gene expression',
                    'protein expression', 'enzyme activity', 'receptor binding',
                    'drug interactions', 'therapeutic effects'
                }
                if entity_clean in generic_phrases:
                    return False
                return True
        
        # Terms with special characters or patterns that indicate specificity
        if any(char in original_entity for char in [':', '/', '\\', '_', '(', ')', '[', ']']):
            return True
        
        # Default: preserve if it doesn't match generic patterns
        return True
    
    def filter_entity_data(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter entity data dictionary (for BioNER results)
        
        Args:
            entity_data: Dictionary mapping entity text to entity info
            
        Returns:
            Filtered dictionary with generic terms removed
        """
        filtered_data = {}
        
        for entity_text, entity_info in entity_data.items():
            if self.should_preserve_entity(entity_text):
                filtered_data[entity_text] = entity_info
            else:
                logger.debug(f"Filtered out generic entity data: '{entity_text}'")
        
        logger.info(f"Filtered entity data: {len(entity_data)} -> {len(filtered_data)} entities")
        return filtered_data
    
    def enhance_biomedical_ner_result(self, ner_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance BioNER result by filtering generic terms
        
        Args:
            ner_result: Result from BioNER tool
            
        Returns:
            Enhanced result with generic terms filtered out
        """
        if 'entities' in ner_result and isinstance(ner_result['entities'], dict):
            # Filter the entities dictionary
            ner_result['entities'] = self.filter_entity_data(ner_result['entities'])
        
        if 'entity_ids' in ner_result and isinstance(ner_result['entity_ids'], dict):
            # Filter the entity_ids dictionary 
            ner_result['entity_ids'] = self.filter_entity_data(ner_result['entity_ids'])
        
        # Add metadata about filtering
        if 'metadata' not in ner_result:
            ner_result['metadata'] = {}
        
        ner_result['metadata']['entity_filtering_applied'] = True
        ner_result['metadata']['filter_version'] = '1.0'
        
        return ner_result


def enhance_bio_ner_with_filtering(bio_ner_tool):
    """
    Enhance existing BioNER tool with intelligent entity filtering
    
    Args:
        bio_ner_tool: Existing BioNER tool instance
        
    Returns:
        Enhanced BioNER tool with filtering capabilities
    """
    entity_filter = BiomedicaleEntityFilter()
    
    # Store original extract_and_link method
    original_extract_and_link = bio_ner_tool.extract_and_link
    
    def enhanced_extract_and_link(query: str, include_types: bool = True) -> Dict[str, Any]:
        """Enhanced extract_and_link with filtering"""
        # Call original method
        result = original_extract_and_link(query, include_types)
        
        # Apply filtering
        enhanced_result = entity_filter.enhance_biomedical_ner_result(result)
        
        return enhanced_result
    
    # Replace method with enhanced version
    bio_ner_tool.extract_and_link = enhanced_extract_and_link
    bio_ner_tool._entity_filter = entity_filter
    
    return bio_ner_tool


# Convenience function for backward compatibility
def filter_generic_biomedical_terms(entities: List[str]) -> List[str]:
    """
    Convenience function to filter generic terms from entity list
    
    Args:
        entities: List of entity strings
        
    Returns:
        Filtered list with generic terms removed
    """
    entity_filter = BiomedicaleEntityFilter()
    return entity_filter.filter_entities(entities)


if __name__ == "__main__":
    # Test the entity filter
    test_entities = [
        "Brucellosis",           # Should preserve - specific disease
        "translation",           # Should preserve - specific biological process  
        "drugs",                 # Should filter - generic term
        "targeting",             # Should filter - generic action word
        "treat",                 # Should filter - generic action word
        "BRCA1",                 # Should preserve - specific gene
        "Type 2 diabetes",       # Should preserve - multi-word specific disease
        "genetic variations",    # Should filter - generic multi-word phrase
        "the",                   # Should filter - generic English word
        "proteins",              # Should filter - generic category
        "aspirin",               # Should preserve - specific drug
        "cholinergic pathway",   # Should preserve - specific pathway
        "TP53",                  # Should preserve - specific gene symbol
        "dopamine",              # Should preserve - specific neurotransmitter
        "protein interactions"   # Should filter - generic biological concept
    ]
    
    entity_filter = BiomedicaleEntityFilter()
    filtered_entities = entity_filter.filter_entities(test_entities)
    
    print("Entity Filtering Test Results:")
    print("=" * 50)
    print(f"Original entities: {test_entities}")
    print(f"Filtered entities: {filtered_entities}")
    print(f"Removed: {set(test_entities) - set(filtered_entities)}")