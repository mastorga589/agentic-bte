#!/usr/bin/env python3
"""
Subquery Placeholder System

This module implements a placeholder system that allows subqueries to use
results from previous subqueries as inputs, enabling proper cascading
biomedical queries.

Key features:
1. Extract specific entities (drug names, gene names) from subquery results
2. Create placeholders like {drugs_from_subquery_1} for use in subsequent queries
3. Resolve placeholders with actual entity IDs when building TRAPI queries
4. Track dependencies between subqueries for proper execution order
5. Enable batched queries using multiple entities from previous results

Example workflow:
- Subquery 1: "What drugs treat Brucellosis?" -> Results: [donepezil, rivastigmine, ...]
- Subquery 2: "What genes do {drugs_from_subquery_1} target?" -> Use actual drug IDs
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class SubqueryPlaceholder:
    """Represents a placeholder for results from a previous subquery"""
    name: str                                    # e.g., "drugs_from_subquery_1"
    source_subquery_index: int                  # Which subquery generated this
    entity_type: str                            # "drug", "gene", "disease", "process"
    entity_ids: List[str] = field(default_factory=list)        # Actual entity IDs
    entity_names: List[str] = field(default_factory=list)      # Human-readable names
    biolink_category: str = "biolink:NamedThing"               # Biolink category
    confidence_threshold: float = 0.3           # Minimum confidence for inclusion


@dataclass 
class SubqueryResult:
    """Represents results from a completed subquery"""
    index: int
    query: str
    entities_used: List[str]
    results: List[Dict[str, Any]]
    success: bool
    confidence: float
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)  # entity_type -> [ids]


class SubqueryPlaceholderSystem:
    """
    System for managing placeholders that carry results between subqueries
    """
    
    def __init__(self):
        """Initialize the placeholder system"""
        self.placeholders: Dict[str, SubqueryPlaceholder] = {}
        self.subquery_results: List[SubqueryResult] = []
        self.dependency_graph: Dict[int, List[int]] = {}  # subquery_index -> [dependent_indices]
        
        # Entity type mappings
        self.entity_type_mappings = {
            'drug': {
                'biolink_category': 'biolink:SmallMolecule',
                'keywords': ['drug', 'medication', 'compound', 'therapeutic', 'treatment'],
                'id_prefixes': ['CHEBI:', 'CHEMBL:', 'DRUGBANK:', 'PUBCHEM:']
            },
            'gene': {
                'biolink_category': 'biolink:Gene', 
                'keywords': ['gene', 'protein', 'polypeptide'],
                'id_prefixes': ['HGNC:', 'ENSEMBL:', 'NCBIGene:', 'UniProtKB:']
            },
            'disease': {
                'biolink_category': 'biolink:Disease',
                'keywords': ['disease', 'disorder', 'condition', 'syndrome'],
                'id_prefixes': ['MONDO:', 'DOID:', 'UMLS:', 'HP:']
            },
            'process': {
                'biolink_category': 'biolink:BiologicalProcess',
                'keywords': ['process', 'pathway', 'function', 'mechanism'],
                'id_prefixes': ['GO:', 'REACT:', 'KEGG:']
            }
        }
    
    def analyze_subquery_for_placeholders(self, query: str, subquery_index: int) -> List[str]:
        """
        Analyze a subquery to identify what placeholders it might need
        
        Args:
            query: Subquery text (e.g., "What genes do these drugs target?")
            subquery_index: Index of this subquery
            
        Returns:
            List of placeholder names that this subquery references
        """
        query_lower = query.lower()
        referenced_placeholders = []
        
        # Look for context-referencing words
        context_references = ['these', 'those', 'identified', 'found', 'discovered']
        
        if any(ref in query_lower for ref in context_references):
            logger.info(f"Subquery {subquery_index} references previous results: {query}")
            
            # Determine what entity type is being referenced
            for entity_type, info in self.entity_type_mappings.items():
                if any(keyword in query_lower for keyword in info['keywords']):
                    # Look for available placeholders of this type
                    for placeholder_name, placeholder in self.placeholders.items():
                        if placeholder.entity_type == entity_type:
                            referenced_placeholders.append(placeholder_name)
                            logger.info(f"  -> References placeholder: {placeholder_name}")
        
        return referenced_placeholders
    
    def extract_entities_from_results(self, subquery_result: SubqueryResult) -> Dict[str, List[str]]:
        """
        Extract specific entities from subquery results by type
        
        Args:
            subquery_result: Completed subquery result
            
        Returns:
            Dictionary mapping entity_type -> [entity_ids]
        """
        extracted = {}
        
        for result in subquery_result.results:
            # Extract from BTE API response format
            if 'knowledge_graph' in result:
                kg = result['knowledge_graph']
                if 'nodes' in kg:
                    for node_id, node_data in kg['nodes'].items():
                        # Determine entity type from biolink category
                        categories = node_data.get('categories', [])
                        entity_name = node_data.get('name', node_id)
                        
                        for category in categories:
                            entity_type = self._biolink_category_to_type(category)
                            if entity_type:
                                if entity_type not in extracted:
                                    extracted[entity_type] = []
                                
                                if node_id not in extracted[entity_type]:
                                    extracted[entity_type].append(node_id)
                                    
                                    # Also try to extract clean names
                                    if entity_name and entity_name != node_id:
                                        extracted[entity_type].append(entity_name)
            
            # Also extract from direct result structure
            for field in ['subject', 'object', 'subject_id', 'object_id']:
                if field in result:
                    entity_id = result[field]
                    if entity_id and ':' in entity_id:  # Looks like a proper ID
                        entity_type = self._id_to_entity_type(entity_id)
                        if entity_type:
                            if entity_type not in extracted:
                                extracted[entity_type] = []
                            if entity_id not in extracted[entity_type]:
                                extracted[entity_type].append(entity_id)
        
        # Remove duplicates and filter by confidence
        for entity_type in extracted:
            extracted[entity_type] = list(set(extracted[entity_type]))
            extracted[entity_type] = extracted[entity_type][:10]  # Limit to top 10 per type
        
        logger.info(f"Extracted entities from subquery {subquery_result.index}: {[(k, len(v)) for k, v in extracted.items()]}")
        return extracted
    
    def create_placeholders_from_subquery(self, subquery_result: SubqueryResult) -> List[str]:
        """
        Create placeholders from a completed subquery's results
        
        Args:
            subquery_result: Completed subquery result
            
        Returns:
            List of placeholder names created
        """
        created_placeholders = []
        
        # Extract entities by type
        extracted_entities = self.extract_entities_from_results(subquery_result)
        subquery_result.extracted_entities = extracted_entities
        
        # Create placeholders for each entity type found
        for entity_type, entity_list in extracted_entities.items():
            if len(entity_list) >= 2:  # Only create placeholder if we have multiple entities
                placeholder_name = f"{entity_type}s_from_subquery_{subquery_result.index + 1}"
                
                # Get biolink category
                biolink_category = self.entity_type_mappings.get(entity_type, {}).get(
                    'biolink_category', 'biolink:NamedThing'
                )
                
                # Separate IDs and names
                entity_ids = [e for e in entity_list if ':' in e and len(e.split(':')) == 2]
                entity_names = [e for e in entity_list if ':' not in e or len(e.split(':')) != 2]
                
                placeholder = SubqueryPlaceholder(
                    name=placeholder_name,
                    source_subquery_index=subquery_result.index,
                    entity_type=entity_type,
                    entity_ids=entity_ids,
                    entity_names=entity_names,
                    biolink_category=biolink_category,
                    confidence_threshold=subquery_result.confidence
                )
                
                self.placeholders[placeholder_name] = placeholder
                created_placeholders.append(placeholder_name)
                
                logger.info(f"Created placeholder: {placeholder_name} with {len(entity_ids)} IDs, {len(entity_names)} names")
        
        return created_placeholders
    
    def resolve_placeholders_in_subquery(self, query: str, subquery_index: int) -> Tuple[str, Dict[str, str]]:
        """
        Resolve placeholders in a subquery to actual entity references
        
        Args:
            query: Subquery text that may contain placeholder references
            subquery_index: Index of the subquery being resolved
            
        Returns:
            Tuple of (resolved_query, enhanced_entity_data)
        """
        resolved_query = query
        enhanced_entity_data = {}
        
        # Find placeholder references in the query
        referenced_placeholders = self.analyze_subquery_for_placeholders(query, subquery_index)
        
        for placeholder_name in referenced_placeholders:
            if placeholder_name in self.placeholders:
                placeholder = self.placeholders[placeholder_name]
                
                # Track dependency
                if subquery_index not in self.dependency_graph:
                    self.dependency_graph[subquery_index] = []
                if placeholder.source_subquery_index not in self.dependency_graph[subquery_index]:
                    self.dependency_graph[subquery_index].append(placeholder.source_subquery_index)
                
                # Add entity data for TRAPI building
                if placeholder.entity_ids:
                    # Use the first few entity IDs for the query
                    key = f"{placeholder.entity_type}s_from_previous"
                    enhanced_entity_data[key] = placeholder.entity_ids[:5]  # Limit to 5 for performance
                    
                    # Mark for category search instead of specific IDs if we have many entities  
                    enhanced_entity_data[f"_category_search_{key}"] = placeholder.biolink_category
                    enhanced_entity_data[f"_batch_entities_{key}"] = "true"
                    
                    logger.info(f"Resolved placeholder {placeholder_name} to {len(placeholder.entity_ids)} entities for subquery {subquery_index}")
        
        return resolved_query, enhanced_entity_data
    
    def _biolink_category_to_type(self, biolink_category: str) -> Optional[str]:
        """Convert biolink category to our entity type"""
        category_mappings = {
            'biolink:SmallMolecule': 'drug',
            'biolink:Drug': 'drug',
            'biolink:Gene': 'gene', 
            'biolink:Protein': 'gene',
            'biolink:Disease': 'disease',
            'biolink:BiologicalProcess': 'process',
            'biolink:PhysiologicalProcess': 'process'
        }
        return category_mappings.get(biolink_category)
    
    def _id_to_entity_type(self, entity_id: str) -> Optional[str]:
        """Determine entity type from entity ID prefix"""
        for entity_type, info in self.entity_type_mappings.items():
            for prefix in info['id_prefixes']:
                if entity_id.startswith(prefix):
                    return entity_type
        return None
    
    def record_subquery_completion(self, subquery_index: int, query: str, 
                                 entities_used: List[str], results: List[Dict[str, Any]], 
                                 success: bool, confidence: float) -> List[str]:
        """
        Record completion of a subquery and create placeholders from results
        
        Args:
            subquery_index: Index of completed subquery
            query: Subquery text
            entities_used: Entities that were used in this subquery
            results: Results from the subquery
            success: Whether the subquery succeeded
            confidence: Average confidence of results
            
        Returns:
            List of placeholder names created from this subquery
        """
        subquery_result = SubqueryResult(
            index=subquery_index,
            query=query,
            entities_used=entities_used,
            results=results,
            success=success,
            confidence=confidence
        )
        
        self.subquery_results.append(subquery_result)
        
        # Create placeholders if subquery was successful and has good results
        created_placeholders = []
        if success and len(results) >= 2:
            created_placeholders = self.create_placeholders_from_subquery(subquery_result)
        
        return created_placeholders
    
    def get_execution_order(self, subqueries: List[str]) -> List[int]:
        """
        Determine optimal execution order based on dependencies
        
        Args:
            subqueries: List of subquery strings
            
        Returns:
            List of subquery indices in dependency-resolved execution order
        """
        # For now, use simple sequential order
        # In a more advanced implementation, this would do topological sort
        return list(range(len(subqueries)))
    
    def get_placeholder_summary(self) -> Dict[str, Any]:
        """
        Get summary of current placeholders for debugging/reporting
        
        Returns:
            Summary dictionary with placeholder information
        """
        summary = {
            'total_placeholders': len(self.placeholders),
            'total_subqueries_completed': len(self.subquery_results),
            'placeholders': {},
            'dependencies': self.dependency_graph
        }
        
        for name, placeholder in self.placeholders.items():
            summary['placeholders'][name] = {
                'entity_type': placeholder.entity_type,
                'entity_count': len(placeholder.entity_ids),
                'source_subquery': placeholder.source_subquery_index,
                'biolink_category': placeholder.biolink_category,
                'sample_entities': placeholder.entity_ids[:3]
            }
        
        return summary
    
    def enhance_entity_data_with_placeholders(self, entity_data: Dict[str, str], 
                                            subquery_index: int) -> Dict[str, str]:
        """
        Enhance entity data with resolved placeholder information for TRAPI building
        
        Args:
            entity_data: Original entity data
            subquery_index: Current subquery index
            
        Returns:
            Enhanced entity data with placeholder resolutions
        """
        enhanced = entity_data.copy()
        
        # Add available placeholders as potential entities
        for placeholder_name, placeholder in self.placeholders.items():
            # Only use placeholders from previous subqueries
            if placeholder.source_subquery_index < subquery_index:
                if placeholder.entity_ids:
                    # Add as batch entity data
                    batch_key = f"batch_{placeholder.entity_type}s"
                    enhanced[batch_key] = placeholder.entity_ids
                    
                    # Add biolink category information
                    enhanced[f"_category_search_{batch_key}"] = placeholder.biolink_category
                    enhanced[f"_batch_mode_{batch_key}"] = "true"
        
        return enhanced


def enhance_production_got_optimizer_with_placeholders(optimizer):
    """
    Enhance the Production GoT Optimizer with placeholder system
    
    Args:
        optimizer: ProductionGoTOptimizer instance
        
    Returns:
        Enhanced optimizer with placeholder capabilities
    """
    # Add placeholder system to optimizer
    optimizer.placeholder_system = SubqueryPlaceholderSystem()
    
    # Store original subquery execution method
    original_execute_subquery = optimizer._execute_subquery_with_parallel_predicates
    
    async def enhanced_execute_subquery(subquery: str, entities: List[Dict[str, Any]], 
                                      subquery_index: int) -> List:
        """Enhanced subquery execution with placeholder resolution"""
        
        # Resolve any placeholders in the subquery
        resolved_query, enhanced_entity_data = optimizer.placeholder_system.resolve_placeholders_in_subquery(
            subquery, subquery_index
        )
        
        # Enhance original entity data with placeholder data
        original_entity_data = {entity.get('name', f'entity_{i}'): entity.get('id', '') 
                              for i, entity in enumerate(entities)}
        
        combined_entity_data = optimizer.placeholder_system.enhance_entity_data_with_placeholders(
            original_entity_data, subquery_index
        )
        combined_entity_data.update(enhanced_entity_data)
        
        logger.info(f"Subquery {subquery_index + 1} enhanced with {len(enhanced_entity_data)} placeholder entities")
        
        # Execute the subquery with enhanced data
        steps = await original_execute_subquery(resolved_query, entities, subquery_index)
        
        # Extract results for placeholder creation
        all_results = []
        total_confidence = 0.0
        success_count = 0
        
        for step in steps:
            if step.step_type == 'api_execution' and step.success:
                step_results = step.output_data.get('results', [])
                all_results.extend(step_results)
                total_confidence += step.confidence
                success_count += 1
        
        # Record subquery completion and create placeholders
        avg_confidence = total_confidence / max(1, success_count)
        created_placeholders = optimizer.placeholder_system.record_subquery_completion(
            subquery_index, resolved_query, [e.get('name') for e in entities],
            all_results, len(all_results) > 0, avg_confidence
        )
        
        if created_placeholders:
            logger.info(f"Created {len(created_placeholders)} placeholders from subquery {subquery_index + 1}: {created_placeholders}")
        
        return steps
    
    # Replace the method with enhanced version
    optimizer._execute_subquery_with_parallel_predicates = enhanced_execute_subquery
    
    return optimizer


# Test and demonstration functions
def demonstrate_placeholder_system():
    """Demonstrate the placeholder system with example data"""
    system = SubqueryPlaceholderSystem()
    
    print("=== Subquery Placeholder System Demo ===")
    
    # Simulate subquery 1 completion: "What drugs treat Brucellosis?"
    mock_results_1 = [
        {
            'knowledge_graph': {
                'nodes': {
                    'CHEBI:27882': {'name': 'donepezil', 'categories': ['biolink:SmallMolecule']},
                    'CHEBI:133011': {'name': 'rivastigmine', 'categories': ['biolink:SmallMolecule']},
                    'CHEBI:17076': {'name': 'galantamine', 'categories': ['biolink:SmallMolecule']}
                }
            }
        }
    ]
    
    placeholders_1 = system.record_subquery_completion(
        0, "What drugs treat Brucellosis?", ['Brucellosis'], mock_results_1, True, 0.8
    )
    print(f"Subquery 1 created placeholders: {placeholders_1}")
    
    # Simulate subquery 2: "What genes do these drugs target?"
    query_2 = "What genes do these drugs target?"
    resolved_query_2, enhanced_data_2 = system.resolve_placeholders_in_subquery(query_2, 1)
    
    print(f"Subquery 2 original: {query_2}")
    print(f"Subquery 2 resolved: {resolved_query_2}")
    print(f"Enhanced entity data: {enhanced_data_2}")
    
    # Show final summary
    summary = system.get_placeholder_summary()
    print(f"Final summary: {summary}")


if __name__ == "__main__":
    demonstrate_placeholder_system()