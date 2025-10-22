#!/usr/bin/env python3
"""
Diagnostic script to test different predicates and data availability in local BTE
"""

import requests
import json
import time
from typing import List, Dict, Any

LOCAL_BTE_URL = "http://localhost:3000/v1"

def test_predicate_variations():
    """Test different predicates to see what works with local BTE"""
    
    print("🧪 TESTING DIFFERENT PREDICATES WITH LOCAL BTE")
    print("=" * 50)
    
    # Common predicates to test
    predicates_to_test = [
        # Basic relationships
        "biolink:related_to",
        "biolink:associated_with",
        
        # Drug-gene relationships
        "biolink:affects",
        "biolink:response_affected_by",
        "biolink:interacts_with",
        
        # More specific
        "biolink:regulates",
        "biolink:positively_regulates", 
        "biolink:negatively_regulates",
        
        # Causal
        "biolink:causes",
        "biolink:contributes_to",
        
        # No predicate (let BTE decide)
        None
    ]
    
    base_query = {
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
    
    results = []
    
    for predicate in predicates_to_test:
        print(f"\n🔍 Testing predicate: {predicate or 'No predicate (BTE decides)'}")
        
        # Copy base query and add predicate if specified
        test_query = json.loads(json.dumps(base_query))  # Deep copy
        if predicate:
            test_query["message"]["query_graph"]["edges"]["e0"]["predicates"] = [predicate]
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{LOCAL_BTE_URL}/query",
                json=test_query,
                headers={"Content-Type": "application/json"},
                timeout=45
            )
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                kg = result.get("message", {}).get("knowledge_graph", {})
                query_results = result.get("message", {}).get("results", [])
                
                nodes_count = len(kg.get("nodes", {}))
                edges_count = len(kg.get("edges", {}))
                results_count = len(query_results)
                
                print(f"   ⏱️  Time: {execution_time:.2f}s")
                print(f"   📊 Nodes: {nodes_count}, Edges: {edges_count}, Results: {results_count}")
                
                if results_count > 0:
                    print(f"   ✅ SUCCESS! Found {results_count} results")
                    
                    # Show sample results
                    if results_count > 0:
                        sample_result = query_results[0]
                        print(f"   📋 Sample result: {sample_result}")
                else:
                    print(f"   ❌ No results")
                
                results.append({
                    'predicate': predicate,
                    'success': results_count > 0,
                    'nodes': nodes_count,
                    'edges': edges_count,
                    'results': results_count,
                    'time': execution_time
                })
                
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ Timeout after 45s")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return results

def test_different_compounds():
    """Test different well-known compounds to see if the issue is aspirin-specific"""
    
    print(f"\n🧬 TESTING DIFFERENT COMPOUNDS")
    print("=" * 35)
    
    # Well-known compounds to test
    compounds = [
        {"name": "Aspirin", "id": "CHEBI:15365"},
        {"name": "Ibuprofen", "id": "CHEBI:5855"}, 
        {"name": "Acetaminophen", "id": "CHEBI:46195"},
        {"name": "Caffeine", "id": "CHEBI:27732"},
        {"name": "Glucose", "id": "CHEBI:17234"},
        {"name": "Insulin", "id": "CHEBI:145810"}  # If it has a CHEBI ID
    ]
    
    for compound in compounds:
        print(f"\n🧪 Testing {compound['name']} ({compound['id']})")
        
        test_query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": [compound['id']],
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
        
        try:
            response = requests.post(
                f"{LOCAL_BTE_URL}/query",
                json=test_query,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                kg = result.get("message", {}).get("knowledge_graph", {})
                query_results = result.get("message", {}).get("results", [])
                
                nodes_count = len(kg.get("nodes", {}))
                edges_count = len(kg.get("edges", {}))
                results_count = len(query_results)
                
                if results_count > 0:
                    print(f"   ✅ SUCCESS: {results_count} results, {nodes_count} nodes, {edges_count} edges")
                    break  # Found one that works!
                else:
                    print(f"   ❌ No results")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def check_bte_data_sources():
    """Check what data sources are available in local BTE"""
    
    print(f"\n📊 CHECKING LOCAL BTE DATA SOURCES")
    print("=" * 40)
    
    try:
        # Check meta knowledge graph for available sources
        response = requests.get(f"{LOCAL_BTE_URL}/meta_knowledge_graph", timeout=10)
        
        if response.status_code == 200:
            meta_kg = response.json()
            edges = meta_kg.get('edges', [])
            
            print(f"✅ Meta-KG has {len(edges)} edges")
            
            # Extract data sources
            sources = set()
            predicates = set()
            
            for edge in edges:
                # Look for API information
                api_info = edge.get('api', {})
                if api_info:
                    api_name = api_info.get('name', 'Unknown')
                    sources.add(api_name)
                
                # Collect predicates
                predicate = edge.get('predicate')
                if predicate:
                    predicates.add(predicate)
            
            print(f"\n🔗 Available Data Sources ({len(sources)}):")
            for source in sorted(sources)[:10]:  # Show first 10
                print(f"   • {source}")
            if len(sources) > 10:
                print(f"   ... and {len(sources) - 10} more")
            
            print(f"\n🔀 Available Predicates ({len(predicates)}):")
            for predicate in sorted(predicates)[:15]:  # Show first 15
                print(f"   • {predicate}")
            if len(predicates) > 15:
                print(f"   ... and {len(predicates) - 15} more")
        else:
            print(f"❌ Could not fetch meta-KG: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking data sources: {e}")

def test_gene_to_drug_direction():
    """Test the reverse direction: Gene to Drug"""
    
    print(f"\n🔄 TESTING REVERSE DIRECTION: GENE TO DRUG")
    print("=" * 45)
    
    # Test with a well-known gene
    test_query = {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {
                        "ids": ["NCBIGene:1956"],  # EGFR gene
                        "categories": ["biolink:Gene"]
                    },
                    "n1": {
                        "categories": ["biolink:SmallMolecule"]
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
    
    print("🧬 Testing EGFR gene (NCBIGene:1956) -> SmallMolecule")
    
    try:
        response = requests.post(
            f"{LOCAL_BTE_URL}/query",
            json=test_query,
            headers={"Content-Type": "application/json"},
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            kg = result.get("message", {}).get("knowledge_graph", {})
            query_results = result.get("message", {}).get("results", [])
            
            nodes_count = len(kg.get("nodes", {}))
            edges_count = len(kg.get("edges", {}))
            results_count = len(query_results)
            
            print(f"✅ Gene->Drug query results:")
            print(f"   📊 Nodes: {nodes_count}, Edges: {edges_count}, Results: {results_count}")
            
            if results_count > 0:
                print("🎉 SUCCESS! Your local BTE has data - the direction matters!")
            else:
                print("❌ Still no results in reverse direction")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all diagnostic tests"""
    
    # Test 1: Different predicates
    predicate_results = test_predicate_variations()
    
    # Test 2: Different compounds
    test_different_compounds()
    
    # Test 3: Data sources
    check_bte_data_sources()
    
    # Test 4: Reverse direction
    test_gene_to_drug_direction()
    
    # Summary
    print(f"\n" + "="*60)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*60)
    
    successful_predicates = [r for r in predicate_results if r['success']]
    
    if successful_predicates:
        print("✅ GOOD NEWS: Found working predicates!")
        for result in successful_predicates:
            print(f"   • {result['predicate']}: {result['results']} results")
    else:
        print("⚠️  No predicates returned results with aspirin->gene")
        print("   This suggests either:")
        print("   1. Local BTE has limited drug-gene data")
        print("   2. Different data source configuration needed")
        print("   3. Different query patterns required")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("   • Check your local BTE configuration and data sources")
    print("   • Try gene-to-drug queries instead")
    print("   • Consider using different compound IDs or gene targets")
    print("   • Check local BTE logs for any errors or warnings")

if __name__ == "__main__":
    main()