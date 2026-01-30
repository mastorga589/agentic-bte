#!/usr/bin/env python3
"""
Diagnostic script to examine entity name resolution issues in final results
"""

import asyncio
import json
import os
import glob
from agentic_bte.core.queries.production_got_optimizer import execute_biomedical_query

async def diagnose_entity_resolution():
    print("🔬 DIAGNOSING ENTITY NAME RESOLUTION ISSUES")
    print("=" * 60)
    
    # Test multiple queries to find one that works
    queries = [
        "What genes are related to aspirin?",
        "What genetic factors influence antipsychotic drug efficacy?",
        "What genes are associated with schizophrenia?"
    ]
    
    for query in queries:
        print(f"\n🎯 Test Query: {query}")
        print("-" * 40)
        
        try:
            result, presentation = await execute_biomedical_query(query)
            
            print("📊 EXECUTION RESULTS:")
            print("-" * 30)
            print(f"Success: {result.success}")
            print(f"Total Results: {result.total_results}")
            print(f"Entities Found: {len(result.entities_found)}")
            
            if result.total_results > 0:
                print(f"\n🔍 EXAMINING FIRST 5 RESULTS FOR ENTITY NAMES:")
                print("-" * 50)
                
                # Access the actual final_results data from the saved file
                # Find the most recent result file
                result_files = glob.glob("got_result_*.json")
                if result_files:
                    latest_file = max(result_files, key=os.path.getctime)
                    print(f"📁 Reading from: {latest_file}")
                    
                    with open(latest_file, 'r') as f:
                        saved_result = json.load(f)
                    
                    final_results = saved_result.get('final_results', [])
                    entities = saved_result.get('entities_found', [])
                    
                    print(f"\n📋 RAW RESULT STRUCTURE (first 3 results):")
                    for i, result_item in enumerate(final_results[:3], 1):
                        print(f"\nResult {i}:")
                        print(f"  Keys: {list(result_item.keys())}")
                        
                        # Check for subject/object names vs IDs
                        subject = result_item.get('subject', 'N/A')
                        subject_id = result_item.get('subject_id', 'N/A')
                        object_name = result_item.get('object', 'N/A')
                        object_id = result_item.get('object_id', 'N/A')
                        predicate = result_item.get('predicate', 'N/A')
                        score = result_item.get('score', 0.0)
                        
                        print(f"  Subject: '{subject}' (ID: {subject_id})")
                        print(f"  Object: '{object_name}' (ID: {object_id})")
                        print(f"  Predicate: {predicate}")
                        print(f"  Score: {score}")
                        
                        # Check if subject/object are actual names or just IDs
                        if subject == subject_id:
                            print(f"  ⚠️  Subject name = ID (not resolved!)")
                        if object_name == object_id:
                            print(f"  ⚠️  Object name = ID (not resolved!)")
                        
                        # Check for knowledge graph data
                        if 'knowledge_graph' in result_item:
                            kg = result_item['knowledge_graph']
                            nodes = kg.get('nodes', {})
                            print(f"  📊 KG Nodes: {len(nodes)}")
                            if nodes:
                                # Show first node example
                                first_node_id, first_node_data = next(iter(nodes.items()))
                                node_name = first_node_data.get('name', 'No name')
                                print(f"  📝 Example node: {first_node_id} -> '{node_name}'")
                    
                    print(f"\n🧬 ENTITY EXTRACTION RESULTS:")
                    print("-" * 35)
                    for i, entity in enumerate(entities[:5], 1):
                        name = entity.get('name', 'Unknown')
                        entity_id = entity.get('id', 'Unknown')
                        entity_type = entity.get('type', 'Unknown')
                        print(f"  {i}. '{name}' (ID: {entity_id}, Type: {entity_type})")
                    
                    print(f"\n🔍 DIAGNOSIS:")
                    print("-" * 20)
                    
                    # Count how many results have proper names vs IDs
                    proper_names = 0
                    id_only = 0
                    
                    for result_item in final_results[:10]:  # Check first 10
                        subject = result_item.get('subject', '')
                        object_name = result_item.get('object', '')
                        subject_id = result_item.get('subject_id', '')
                        object_id = result_item.get('object_id', '')
                        
                        if subject != subject_id and object_name != object_id:
                            proper_names += 1
                        elif subject == subject_id or object_name == object_id:
                            id_only += 1
                    
                    print(f"Results with proper names: {proper_names}/10")
                    print(f"Results with only IDs: {id_only}/10")
                    
                    if id_only > proper_names:
                        print("❌ ISSUE: Most results show IDs instead of human-readable names!")
                        print("   This indicates entity name resolution is failing.")
                    elif proper_names > 5:
                        print("✅ GOOD: Most results have proper entity names")
                    else:
                        print("⚠️  MIXED: Some results have names, others have IDs")
                    
                    # Check for generic entities like "genetic variations"
                    generic_terms = ["genetic variations", "genetic factors", "drug metabolism", "dopamine receptor pathways"]
                    generic_found = []
                    
                    for result_item in final_results:
                        subject = result_item.get('subject', '').lower()
                        object_name = result_item.get('object', '').lower()
                        
                        for term in generic_terms:
                            if term in subject or term in object_name:
                                generic_found.append(term)
                    
                    if generic_found:
                        print(f"\n⚠️  GENERIC ENTITY ISSUE:")
                        print(f"   Found generic terms: {list(set(generic_found))}")
                        print("   These should be specific gene/drug names from the knowledge graph")
                    
                    # If we found results, we can analyze this query and break
                    print(f"\n✅ Found working query: {query}")
                    break
                    
                else:
                    print("❌ No result files found to examine")
            else:
                print("❌ No results to examine")
                
        except Exception as e:
            print(f"❌ Error in diagnosis: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose_entity_resolution())