#!/usr/bin/env python3
"""
Investigate entity names in knowledge graph nodes from existing results
"""

import json
import glob
import os

def investigate_kg_nodes():
    print("🔍 INVESTIGATING KNOWLEDGE GRAPH NODE NAMES")
    print("=" * 60)
    
    # Find the most recent result file
    result_files = glob.glob("got_result_*.json")
    if not result_files:
        print("❌ No result files found")
        return
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"📁 Reading from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        saved_result = json.load(f)
    
    final_results = saved_result.get('final_results', [])
    print(f"📊 Examining {len(final_results)} results for KG node names...")
    
    # Look for knowledge graph nodes with names
    umls_entities_found = {}
    total_nodes_examined = 0
    nodes_with_names = 0
    
    for i, result in enumerate(final_results[:10]):  # Check first 10 results
        print(f"\n📋 Result {i+1}:")
        
        # Check if this result has knowledge graph data
        kg = result.get('knowledge_graph', {})
        if not kg:
            print(f"   ❌ No knowledge_graph in result {i+1}")
            continue
        
        nodes = kg.get('nodes', {})
        edges = kg.get('edges', {})
        
        print(f"   📊 KG has {len(nodes)} nodes, {len(edges)} edges")
        
        # Examine nodes for names
        for node_id, node_data in nodes.items():
            total_nodes_examined += 1
            
            name = node_data.get('name')
            categories = node_data.get('categories', [])
            
            if name:
                nodes_with_names += 1
                print(f"   ✅ Node {node_id}: '{name}' (categories: {categories})")
                
                # If this is a UMLS ID, record the mapping
                if node_id.startswith('UMLS:'):
                    umls_entities_found[node_id] = name
            else:
                print(f"   ❌ Node {node_id}: NO NAME (categories: {categories})")
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total nodes examined: {total_nodes_examined}")
    print(f"   Nodes with names: {nodes_with_names}")
    print(f"   UMLS entities with names: {len(umls_entities_found)}")
    
    if umls_entities_found:
        print(f"\n✅ UMLS ENTITIES FOUND WITH NAMES:")
        for umls_id, name in list(umls_entities_found.items())[:10]:
            print(f"   {umls_id} -> '{name}'")
    
    # Check final result structure for direct name extraction
    print(f"\n🔍 CHECKING RESULT STRUCTURE FOR ENTITY NAMES:")
    for i, result in enumerate(final_results[:3]):
        print(f"\nResult {i+1} structure:")
        print(f"   Keys: {list(result.keys())}")
        
        subject = result.get('subject', 'N/A')
        subject_id = result.get('subject_id', 'N/A')
        object_name = result.get('object', 'N/A')
        object_id = result.get('object_id', 'N/A')
        
        print(f"   Subject: '{subject}' (ID: {subject_id})")
        print(f"   Object: '{object_name}' (ID: {object_id})")
        
        # Check if there's a mismatch between names and IDs
        if subject == subject_id and subject_id.startswith('UMLS:'):
            # Check if KG has name for this ID
            kg = result.get('knowledge_graph', {})
            nodes = kg.get('nodes', {})
            if subject_id in nodes and nodes[subject_id].get('name'):
                actual_name = nodes[subject_id]['name']
                print(f"   ⚠️  Subject name missing! KG has: '{actual_name}'")
        
        if object_name == object_id and object_id.startswith('UMLS:'):
            # Check if KG has name for this ID
            kg = result.get('knowledge_graph', {})
            nodes = kg.get('nodes', {})
            if object_id in nodes and nodes[object_id].get('name'):
                actual_name = nodes[object_id]['name']
                print(f"   ⚠️  Object name missing! KG has: '{actual_name}'")

if __name__ == "__main__":
    investigate_kg_nodes()