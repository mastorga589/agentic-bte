#!/usr/bin/env python3
"""
Detailed test script to inspect TRAPI queries sent to local BTE instance
"""

import asyncio
import json
import requests
import time
from typing import Dict, Any
from agentic_bte.core.queries.production_got_optimizer import execute_biomedical_query

async def test_with_trapi_inspection():
    print("🔬 DETAILED LOCAL BTE TEST WITH TRAPI INSPECTION")
    print("=" * 60)
    
    # Test query
    query = "What genes are related to aspirin?"
    print(f"🎯 Query: {query}")
    print()
    
    # First, let's manually create and inspect a TRAPI query
    print("📋 STEP 1: MANUAL TRAPI QUERY CREATION")
    print("-" * 40)
    
    try:
        from agentic_bte.core.queries.mcp_integration import call_mcp_tool
        
        # Extract entities first
        print("🧬 Extracting entities...")
        entities_response = await call_mcp_tool("bio_ner", query=query)
        entities = entities_response.get('entities', [])
        print(f"✅ Found {len(entities)} entities:")
        for i, entity in enumerate(entities, 1):
            print(f"   {i}. {entity.get('name', 'Unknown')} (Type: {entity.get('type', 'N/A')}, ID: {entity.get('id', 'N/A')})")
        
        # Build TRAPI query
        print(f"\n🔧 Building TRAPI query...")
        entity_names = [e.get('name', '') for e in entities]
        trapi_response = await call_mcp_tool("build_trapi_query", query=query, entity_data={})
        
        trapi_query = trapi_response.get('query', {})
        print("✅ TRAPI query generated successfully!")
        
        # Display the TRAPI query structure
        print(f"\n📊 GENERATED TRAPI QUERY STRUCTURE:")
        print("=" * 50)
        print(json.dumps(trapi_query, indent=2))
        
        # Test the query directly against local BTE
        print(f"\n🚀 STEP 2: DIRECT BTE API TEST")
        print("-" * 30)
        
        local_bte_url = "http://localhost:3000/v1/query"
        print(f"📡 Testing direct API call to: {local_bte_url}")
        
        start_time = time.time()
        try:
            response = requests.post(
                local_bte_url,
                json=trapi_query,
                headers={"Content-Type": "application/json"},
                timeout=60  # Longer timeout for local testing
            )
            
            execution_time = time.time() - start_time
            print(f"⏱️  Response time: {execution_time:.2f} seconds")
            print(f"📋 HTTP Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Analyze the response
                message = result.get("message", {})
                knowledge_graph = message.get("knowledge_graph", {})
                query_graph = message.get("query_graph", {})
                results = message.get("results", [])
                
                nodes_count = len(knowledge_graph.get("nodes", {}))
                edges_count = len(knowledge_graph.get("edges", {}))
                results_count = len(results)
                
                print(f"✅ TRAPI Response Analysis:")
                print(f"   • Knowledge Graph Nodes: {nodes_count}")
                print(f"   • Knowledge Graph Edges: {edges_count}")
                print(f"   • Results: {results_count}")
                
                # Show some sample results if available
                if results:
                    print(f"\n📋 SAMPLE RESULTS (showing first 3):")
                    for i, result in enumerate(results[:3], 1):
                        node_bindings = result.get("node_bindings", {})
                        edge_bindings = result.get("edge_bindings", {})
                        print(f"   Result {i}:")
                        print(f"     Nodes: {list(node_bindings.keys())}")
                        print(f"     Edges: {list(edge_bindings.keys())}")
                
                # Show some sample knowledge graph nodes
                if knowledge_graph.get("nodes"):
                    print(f"\n🧬 SAMPLE KNOWLEDGE GRAPH NODES (first 3):")
                    sample_nodes = list(knowledge_graph["nodes"].items())[:3]
                    for node_id, node_data in sample_nodes:
                        name = node_data.get("name", "Unknown")
                        categories = node_data.get("categories", [])
                        print(f"   • {node_id}: {name} ({', '.join(categories)})")
                
                # Show some sample edges
                if knowledge_graph.get("edges"):
                    print(f"\n🔗 SAMPLE KNOWLEDGE GRAPH EDGES (first 3):")
                    sample_edges = list(knowledge_graph["edges"].items())[:3]
                    for edge_id, edge_data in sample_edges:
                        subject = edge_data.get("subject", "Unknown")
                        predicate = edge_data.get("predicate", "Unknown")
                        object_node = edge_data.get("object", "Unknown")
                        print(f"   • {edge_id}: {subject} --{predicate}--> {object_node}")
                
            else:
                print(f"❌ API call failed with status {response.status_code}")
                print(f"Response: {response.text[:500]}...")
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out after 60 seconds")
        except Exception as e:
            print(f"❌ API call error: {e}")
    
    except Exception as e:
        print(f"❌ TRAPI generation error: {e}")
        import traceback
        traceback.print_exc()
    
    # Now test with the full agentic-bte pipeline
    print(f"\n🔬 STEP 3: FULL AGENTIC-BTE PIPELINE TEST")
    print("-" * 40)
    
    try:
        print("⚡ Executing full pipeline with local BTE...")
        start_time = time.time()
        
        result, presentation = await execute_biomedical_query(query)
        
        execution_time = time.time() - start_time
        print(f"⏱️  Full pipeline execution time: {execution_time:.2f} seconds")
        print(f"✅ Pipeline Success: {result.success}")
        print(f"📊 Results found: {result.total_results}")
        print(f"🧠 Entities found: {len(result.entities_found)}")
        print(f"📈 Quality score: {result.quality_score:.3f}")
        
        if result.final_answer:
            print(f"\n📋 FINAL ANSWER PREVIEW:")
            preview_length = 400
            answer_preview = result.final_answer[:preview_length]
            if len(result.final_answer) > preview_length:
                answer_preview += "..."
            print(answer_preview)
        
    except Exception as e:
        print(f"❌ Full pipeline error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("🎯 TEST COMPLETE")

def test_simple_direct_query():
    """Test a simple direct query to local BTE"""
    print("\n🧪 BONUS: SIMPLE DIRECT QUERY TEST")
    print("-" * 35)
    
    # Very simple TRAPI query for testing
    simple_query = {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {
                        "ids": ["CHEBI:15365"],  # Aspirin
                        "categories": ["biolink:SmallMolecule"]
                    },
                    "n1": {
                        "categories": ["biolink:Gene"]
                    }
                },
                "edges": {
                    "e0": {
                        "subject": "n0",
                        "object": "n1"
                    }
                }
            }
        }
    }
    
    print("🔬 Testing very simple TRAPI query:")
    print("   Query: Aspirin (CHEBI:15365) -> Any Gene")
    print(json.dumps(simple_query, indent=2))
    
    try:
        response = requests.post(
            "http://localhost:3000/v1/query",
            json=simple_query,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            kg = result.get("message", {}).get("knowledge_graph", {})
            results = result.get("message", {}).get("results", [])
            
            print(f"✅ Simple query successful!")
            print(f"   Nodes: {len(kg.get('nodes', {}))}")
            print(f"   Edges: {len(kg.get('edges', {}))}")
            print(f"   Results: {len(results)}")
        else:
            print(f"❌ Simple query failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Simple query error: {e}")

if __name__ == "__main__":
    asyncio.run(test_with_trapi_inspection())
    test_simple_direct_query()