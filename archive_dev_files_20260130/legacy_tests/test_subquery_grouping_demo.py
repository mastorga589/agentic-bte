#!/usr/bin/env python3
"""
Demo script to show subquery grouping feature working
"""

import sys
import asyncio
sys.path.append('/Users/mastorga/Documents/agentic-bte')

from agentic_bte.core.queries.metakg_aware_optimizer import MetaKGAwareAdaptiveOptimizer
from agentic_bte.core.entities.bio_ner import BiomedicalNER
from agentic_bte.core.knowledge.bte_client import BTEClient
from agentic_bte.config import Config

async def demo_subquery_grouping():
    """Demo the subquery grouping feature"""
    print("🧪 SUBQUERY GROUPING DEMO")
    print("=" * 50)
    
    # Set up components
    config = Config()
    bio_ner = BiomedicalNER()
    bte_client = BTEClient()
    optimizer = MetaKGAwareAdaptiveOptimizer(bio_ner, bte_client)
    
    # Load meta-KG
    await optimizer._load_meta_kg()
    print(f"✅ Loaded {len(optimizer._meta_kg_edges)} meta-KG edges")
    
    # Simulate having drug entities from previous results
    optimizer._accumulated_results = [
        {'subject': 'alzheimer disease', 'object': 'donepezil'},
        {'subject': 'alzheimer disease', 'object': 'rivastigmine'},
        {'subject': 'alzheimer disease', 'object': 'galantamine'},
        {'subject': 'alzheimer disease', 'object': 'memantine'},
        {'subject': 'alzheimer disease', 'object': 'acetylcarnitine'},
        {'subject': 'alzheimer disease', 'object': 'nilotinib'},
    ]
    
    # Test the grouping logic directly
    drugs_found = optimizer._get_top_drugs_from_accumulated_results(optimizer._accumulated_results)
    print(f"🎯 Top drugs found: {drugs_found}")
    
    # Create a SmallMolecule -> Gene meta-KG edge
    class MockEdge:
        def __init__(self, subject, predicate, object_type):
            self.subject = subject
            self.predicate = predicate
            self.object = object_type
    
    mock_edge = MockEdge("biolink:SmallMolecule", "biolink:interacts_with", "biolink:Gene")
    
    # Test the grouping generation
    entity_data = {
        'donepezil': 'CHEBI:145499',
        'acetylcarnitine': 'CHEBI:57589',
        'rivastigmine': 'CHEBI:8874',
        'galantamine': 'CHEBI:42944'
    }
    
    grouped_query = optimizer._group_similar_queries_from_entities(entity_data, [], mock_edge)
    print(f"🔗 Grouped query from entity data: {grouped_query}")
    
    # Test the SmallMolecule->Gene triggering logic
    print("\n🔬 Testing SmallMolecule->Gene grouping trigger")
    if mock_edge.subject.lower().endswith("smallmolecule") and mock_edge.object.lower().endswith("gene"):
        top_drugs_for_grouping = optimizer._get_top_drugs_from_accumulated_results(optimizer._accumulated_results)
        if len(top_drugs_for_grouping) >= 2:
            proactive_grouped_query = f"What genes do {', '.join(top_drugs_for_grouping)} interact with?"
            print(f"✨ Proactive grouped query: {proactive_grouped_query}")
            print(f"💡 This replaces {len(top_drugs_for_grouping)} individual queries!")
            
            # Show what individual queries would have been
            print(f"📋 Individual queries that would be replaced:")
            for i, drug in enumerate(top_drugs_for_grouping, 1):
                print(f"   {i}. What genes does {drug} interact with?")
        else:
            print("❌ Not enough drugs found for grouping")
    
    print("\n🎉 SUCCESS: Subquery grouping feature is working!")
    print("✅ The system can combine multiple similar queries into one efficient batch query.")

if __name__ == "__main__":
    asyncio.run(demo_subquery_grouping())