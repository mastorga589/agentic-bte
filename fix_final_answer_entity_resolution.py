#!/usr/bin/env python3
"""
Fix for entity name resolution in final answer generation

The issue is in _prepare_answer_context where entity_mappings only includes
the original entities from the query, but doesn't include the UMLS IDs
found in the BTE results with their proper names from the knowledge graph.

This script shows how to properly extract entity names from BTE results.
"""

import json
import re
from typing import Dict, List, Any

def extract_entity_names_from_results(final_results: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Extract entity names from BTE results by examining knowledge graph nodes
    
    Args:
        final_results: List of BTE result dictionaries
        
    Returns:
        Dictionary mapping entity IDs to their names
    """
    entity_mappings = {}
    
    for result in final_results:
        # Check if this result has knowledge graph data
        kg = result.get('knowledge_graph', {})
        if not kg:
            continue
            
        nodes = kg.get('nodes', {})
        
        # Extract names from all nodes
        for node_id, node_data in nodes.items():
            name = node_data.get('name')
            if name and node_id:
                entity_mappings[node_id] = name
                
                # Also create reverse mapping
                entity_mappings[name] = node_id
    
    return entity_mappings

def demonstrate_fix():
    """
    Demonstrate how the entity name extraction should work
    """
    print("🔧 DEMONSTRATING ENTITY NAME EXTRACTION FIX")
    print("=" * 60)
    
    # Simulate what BTE results might look like (based on test fixtures)
    sample_bte_results = [
        {
            "subject": "MONDO:0005015",
            "subject_id": "MONDO:0005015", 
            "object": "CHEBI:6801",
            "object_id": "CHEBI:6801",
            "predicate": "biolink:treated_by",
            "score": 0.85,
            "knowledge_graph": {
                "nodes": {
                    "MONDO:0005015": {
                        "categories": ["biolink:Disease"],
                        "name": "diabetes mellitus"
                    },
                    "CHEBI:6801": {
                        "categories": ["biolink:SmallMolecule"],
                        "name": "Metformin"
                    }
                },
                "edges": {
                    "edge_1": {
                        "subject": "MONDO:0005015",
                        "predicate": "biolink:treated_by",
                        "object": "CHEBI:6801"
                    }
                }
            }
        },
        {
            "subject": "UMLS:C0018270", 
            "subject_id": "UMLS:C0018270",
            "object": "UMLS:C0162638",
            "object_id": "UMLS:C0162638", 
            "predicate": "biolink:affects",
            "score": 0.29,
            "knowledge_graph": {
                "nodes": {
                    "UMLS:C0018270": {
                        "categories": ["biolink:Gene"],
                        "name": "Growth Factors"
                    },
                    "UMLS:C0162638": {
                        "categories": ["biolink:BiologicalProcess"],
                        "name": "Apoptotic Process"
                    }
                },
                "edges": {
                    "edge_2": {
                        "subject": "UMLS:C0018270",
                        "predicate": "biolink:affects",
                        "object": "UMLS:C0162638"
                    }
                }
            }
        }
    ]
    
    print("📋 Sample BTE Results Structure:")
    for i, result in enumerate(sample_bte_results, 1):
        subject = result.get('subject')
        obj = result.get('object')
        predicate = result.get('predicate')
        
        print(f"\n  Result {i}:")
        print(f"    Subject: {subject}")
        print(f"    Object: {obj}")
        print(f"    Predicate: {predicate}")
        
        # Show what happens WITHOUT the fix
        print(f"    ❌ Current: {subject} → {predicate} → {obj}")
        
    print(f"\n🔍 EXTRACTING ENTITY NAMES FROM KNOWLEDGE GRAPH:")
    print("-" * 50)
    
    # Apply the fix
    entity_mappings = extract_entity_names_from_results(sample_bte_results)
    
    print(f"Found {len(entity_mappings)} entity mappings:")
    for entity_id, name in entity_mappings.items():
        if entity_id.startswith(('UMLS:', 'MONDO:', 'CHEBI:')):
            print(f"  {entity_id} → '{name}'")
    
    print(f"\n✅ FIXED RELATIONSHIPS:")
    print("-" * 30)
    
    # Show what it would look like WITH the fix
    for i, result in enumerate(sample_bte_results, 1):
        subject_id = result.get('subject')
        object_id = result.get('object')
        predicate = result.get('predicate')
        
        # Resolve names using the mappings
        subject_name = entity_mappings.get(subject_id, subject_id)
        object_name = entity_mappings.get(object_id, object_id) 
        
        print(f"  Result {i}: {subject_name} → {predicate} → {object_name}")
    
    print(f"\n🚀 IMPLEMENTATION NEEDED:")
    print("=" * 40)
    print("1. Modify _prepare_answer_context in final_answer_llm.py")
    print("2. Add extraction of entity names from ALL final_results")
    print("3. Update entity_mappings with KG node names")
    print("4. This will resolve UMLS IDs to human-readable names")
    print("5. Final answers will show 'Metformin' instead of 'CHEBI:6801'")

if __name__ == "__main__":
    demonstrate_fix()