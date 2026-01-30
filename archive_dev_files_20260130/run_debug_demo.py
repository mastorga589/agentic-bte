#!/usr/bin/env python3
"""
Simple Test Runner for Enhanced GoT System Debugging

This script provides an easy way to run the debugging demonstration
with different levels of detail based on your needs.
"""

import asyncio
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_menu():
    """Print the menu options"""
    print("🔬 ENHANCED GOT SYSTEM - DEBUG DEMO MENU")
    print("=" * 45)
    print("1. Full System Debugging (Comprehensive)")
    print("2. Individual Component Testing")
    print("3. Quick System Test (Basic)")
    print("4. Entity Extraction Test Only") 
    print("5. TRAPI Query Building Test Only")
    print("6. BTE API Test Only")
    print("0. Exit")
    print()

async def quick_system_test():
    """Quick system test with basic debugging"""
    from debug_enhanced_got_demo import demonstrate_enhanced_got_with_debugging
    
    print("🚀 QUICK SYSTEM TEST")
    print("=" * 25)
    print("Running basic system test with the Brucellosis query...")
    print()
    
    try:
        await demonstrate_enhanced_got_with_debugging()
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()

async def individual_component_test():
    """Test individual MCP components"""
    from debug_enhanced_got_demo import demonstrate_step_by_step_debugging
    
    print("🔧 INDIVIDUAL COMPONENT TEST")
    print("=" * 30)
    print("Testing each MCP tool individually...")
    print()
    
    try:
        await demonstrate_step_by_step_debugging()
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()

async def entity_extraction_test():
    """Test only entity extraction"""
    from agentic_bte.core.queries.mcp_integration import call_mcp_tool
    
    print("🧬 ENTITY EXTRACTION TEST")
    print("=" * 28)
    
    query = "What drugs can treat Brucellosis by targeting translation?"
    print(f"Query: \"{query}\"")
    print()
    
    try:
        print("Calling bio_ner MCP tool...")
        response = await call_mcp_tool("bio_ner", query=query)
        
        print("✅ Entity extraction successful!")
        print(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
        
        if 'entities' in response:
            entities = response['entities']
            print(f"Entities found: {len(entities)}")
            
            for i, entity in enumerate(entities[:5], 1):
                print(f"  {i}. {entity.get('name', 'No name')} ({entity.get('type', 'No type')})")
                print(f"     ID: {entity.get('id', 'No ID')}")
        else:
            print("⚠️  No entities found in response")
            
    except Exception as e:
        print(f"❌ Entity extraction failed: {e}")

async def trapi_building_test():
    """Test only TRAPI query building"""
    from agentic_bte.core.queries.mcp_integration import call_mcp_tool
    
    print("📋 TRAPI QUERY BUILDING TEST")
    print("=" * 30)
    
    query = "What drugs can treat Brucellosis by targeting translation?"
    entity_data = {"Brucellosis": "MONDO:0005683", "drugs": "biolink:SmallMolecule"}
    
    print(f"Query: \"{query}\"")
    print(f"Entity data: {entity_data}")
    print()
    
    try:
        print("Calling build_trapi_query MCP tool...")
        response = await call_mcp_tool(
            "build_trapi_query",
            query=query,
            entity_data=entity_data
        )
        
        print("✅ TRAPI building successful!")
        print(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
        
        if 'query' in response:
            trapi_query = response['query']
            if trapi_query and 'message' in trapi_query:
                message = trapi_query['message']
                if 'query_graph' in message:
                    qg = message['query_graph']
                    nodes = qg.get('nodes', {})
                    edges = qg.get('edges', {})
                    
                    print(f"Query graph created with {len(nodes)} nodes and {len(edges)} edges")
                    
                    for node_id, node_data in nodes.items():
                        categories = node_data.get('categories', [])
                        print(f"  Node {node_id}: {categories}")
        else:
            print("⚠️  No query found in response")
            
    except Exception as e:
        print(f"❌ TRAPI building failed: {e}")

async def bte_api_test():
    """Test only BTE API call"""
    from agentic_bte.core.queries.mcp_integration import call_mcp_tool
    
    print("🔗 BTE API TEST")
    print("=" * 15)
    
    # Simple TRAPI query for testing
    trapi_query = {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"categories": ["biolink:Disease"], "ids": ["MONDO:0005683"]},  # Brucellosis
                    "n1": {"categories": ["biolink:SmallMolecule"]}
                },
                "edges": {
                    "e0": {"subject": "n1", "object": "n0", "predicates": ["biolink:treats"]}
                }
            }
        }
    }
    
    print("Using simple TRAPI query: Disease ← treats ← SmallMolecule")
    print("Disease: MONDO:0005683 (Brucellosis)")
    print()
    
    try:
        print("Calling call_bte_api MCP tool...")
        response = await call_mcp_tool(
            "call_bte_api",
            json_query=trapi_query,
            k=5,
            maxresults=10
        )
        
        print("✅ BTE API call successful!")
        print(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
        
        if 'results' in response:
            results = response['results']
            print(f"Results found: {len(results)}")
            
            for i, result in enumerate(results[:3], 1):
                subject = result.get('subject', 'No subject')
                predicate = result.get('predicate', 'No predicate')
                obj = result.get('object', 'No object')
                score = result.get('score', 'No score')
                
                print(f"  {i}. {subject} → {predicate} → {obj} (score: {score})")
        else:
            print("⚠️  No results found in response")
            
    except Exception as e:
        print(f"❌ BTE API call failed: {e}")

async def main():
    """Main menu loop"""
    while True:
        print_menu()
        choice = input("Select an option (0-6): ").strip()
        print()
        
        try:
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                from debug_enhanced_got_demo import demonstrate_enhanced_got_with_debugging
                await demonstrate_enhanced_got_with_debugging()
            elif choice == "2":
                await individual_component_test()
            elif choice == "3":
                await quick_system_test()
            elif choice == "4":
                await entity_extraction_test()
            elif choice == "5":
                await trapi_building_test()
            elif choice == "6":
                await bte_api_test()
            else:
                print("❌ Invalid choice. Please select 0-6.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n⏸️  Operation cancelled by user")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50)
        input("Press Enter to continue...")
        print("\n" * 2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)