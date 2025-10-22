#!/usr/bin/env python3
"""
Investigate why the system uses generic entities like "genetic variations"
instead of finding specific gene entities from the knowledge graph
"""

import json

def analyze_entity_extraction_issue():
    print("🔍 ANALYZING ENTITY EXTRACTION SPECIFICITY ISSUE")
    print("=" * 60)
    
    # Load the result file to understand the issue
    with open("got_result_20251001_095036.json", 'r') as f:
        result = json.load(f)
    
    print("📋 ORIGINAL QUERY:")
    print(f"   {result['query']}")
    
    print(f"\n🧬 EXTRACTED ENTITIES:")
    print("-" * 30)
    for entity in result['entities_found']:
        name = entity['name']
        entity_type = entity['type'] 
        entity_id = entity['id']
        print(f"  • '{name}' (Type: {entity_type}, ID: {entity_id})")
    
    print(f"\n❓ THE PROBLEM:")
    print(f"   The system extracted 'genetic variations' as a generic entity")
    print(f"   but didn't find SPECIFIC genes like 'DRD2', 'CYP2D6', etc.")
    print(f"   This means TRAPI queries search for the generic concept")
    print(f"   instead of connecting specific genes to drugs/pathways")
    
    print(f"\n🎯 ROOT CAUSE ANALYSIS:")
    print("-" * 25)
    print("1. Entity Extraction Problem:")
    print("   - BioNER extracts 'genetic variations' from user query")
    print("   - But doesn't identify SPECIFIC gene names to search for") 
    print("   - Maps to generic UMLS concept instead of biolink:Gene entities")
    
    print(f"\n2. TRAPI Query Building Problem:")
    print("   - TRAPI queries use generic 'genetic variations' as subject")
    print("   - Should identify that user wants specific GENES")
    print("   - Should build queries like: biolink:Gene -> affects -> biolink:SmallMolecule")
    
    print(f"\n3. Knowledge Graph Search Problem:")
    print("   - BTE searches using generic concept as starting point")
    print("   - Returns relationships involving 'genetic variations' concept") 
    print("   - Instead of returning specific gene-drug relationships")
    
    print(f"\n💡 SOLUTIONS NEEDED:")
    print("=" * 20)
    print("Option 1: Enhanced Entity Recognition")
    print("   - Recognize when user asks about 'genetic factors/variations'")
    print("   - Automatically expand to search for biolink:Gene entities")
    print("   - Let BTE find specific genes related to the query context")
    
    print(f"\nOption 2: Query Intent Understanding")
    print("   - Detect that 'genetic variations' means 'specific genes'")
    print("   - Build TRAPI queries that search for Gene entities")
    print("   - Use open-ended Gene category instead of specific IDs")
    
    print(f"\nOption 3: Two-Stage Approach")
    print("   - First: Find specific genes related to the disease/drug context")
    print("   - Second: Query relationships between those specific genes")
    
    print(f"\n🔧 RECOMMENDED FIX:")
    print("-" * 20)
    print("Modify the TRAPI query building to:")
    print("1. Recognize generic terms like 'genetic variations', 'genetic factors'")
    print("2. Convert them to biolink:Gene category searches")  
    print("3. Let the knowledge graph return specific gene entities")
    print("4. This will show 'DRD2 -> affects -> Haloperidol' instead of")
    print("   'Genetic variations -> affects -> UMLS:C0018270'")

if __name__ == "__main__":
    analyze_entity_extraction_issue()