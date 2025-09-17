#!/usr/bin/env python3
"""
Basic MCP Server Usage Example

This script demonstrates how to use the Agentic BTE MCP tools programmatically
for biomedical entity extraction, TRAPI query building, and BTE API calls.

Usage:
    python examples/mcp_basic_usage.py
"""

import asyncio
import os
from agentic_bte.servers.mcp.tools.bio_ner_tool import handle_bio_ner
from agentic_bte.servers.mcp.tools.trapi_tool import handle_trapi_query  
from agentic_bte.servers.mcp.tools.bte_tool import handle_bte_call
from agentic_bte.servers.mcp.tools.query_tool import handle_plan_and_execute


async def demo_bio_ner():
    """Demonstrate biomedical entity recognition"""
    print("🧬 BioNER Example")
    print("=" * 50)
    
    query = "What drugs can treat type 2 diabetes mellitus?"
    print(f"Query: {query}")
    
    result = await handle_bio_ner({"query": query})
    
    if "error" not in result:
        print("\n✅ Entities extracted successfully!")
        entities = result.get("result_data", {}).get("entities", {})
        for entity_name, entity_data in entities.items():
            print(f"  • {entity_name}: {entity_data.get('id', 'Unknown')} ({entity_data.get('type', 'Unknown')})")
    else:
        print(f"❌ Error: {result['error']}")
    
    return result


async def demo_trapi_building(entity_data):
    """Demonstrate TRAPI query building"""
    print("\n🔧 TRAPI Query Building Example") 
    print("=" * 50)
    
    query = "What drugs can treat diabetes?"
    print(f"Query: {query}")
    print(f"Using entities: {list(entity_data.keys()) if entity_data else 'None'}")
    
    result = await handle_trapi_query({
        "query": query,
        "entity_data": entity_data
    })
    
    if "error" not in result:
        print("\n✅ TRAPI query built successfully!")
        trapi = result.get("trapi_query", {})
        nodes = trapi.get("message", {}).get("query_graph", {}).get("nodes", {})
        edges = trapi.get("message", {}).get("query_graph", {}).get("edges", {})
        print(f"  • Nodes: {len(nodes)}")
        print(f"  • Edges: {len(edges)}")
        
        # Show structure
        for node_id, node_data in nodes.items():
            categories = node_data.get("categories", [])
            ids = node_data.get("ids", [])
            print(f"    - {node_id}: {categories} {ids if ids else ''}")
    else:
        print(f"❌ Error: {result['error']}")
    
    return result


async def demo_bte_api(trapi_query):
    """Demonstrate BTE API execution"""
    print("\n🌐 BTE API Example")
    print("=" * 50)
    
    print("Executing TRAPI query against BTE knowledge graph...")
    
    result = await handle_bte_call({
        "json_query": trapi_query,
        "maxresults": 10,
        "k": 3
    })
    
    if "error" not in result:
        results = result.get("results", [])
        entity_mappings = result.get("entity_mappings", {})
        
        print(f"\n✅ BTE query executed successfully!")
        print(f"  • Found {len(results)} relationships")
        print(f"  • Resolved {len(entity_mappings)} entity names")
        
        # Show sample results
        if results:
            print("\n📊 Sample Results:")
            for i, rel in enumerate(results[:3]):
                subject = rel.get("subject", "Unknown")
                predicate = rel.get("predicate", "unknown_relation")
                obj = rel.get("object", "Unknown")
                clean_predicate = predicate.replace("biolink:", "").replace("_", " ")
                print(f"    {i+1}. {subject} --{clean_predicate}--> {obj}")
        
        # Show entity mappings
        if entity_mappings:
            print(f"\n🏷️ Entity Names:")
            for name, entity_id in list(entity_mappings.items())[:5]:
                print(f"    • {entity_id}: {name}")
    else:
        print(f"❌ Error: {result['error']}")
    
    return result


async def demo_end_to_end():
    """Demonstrate end-to-end biomedical query processing"""
    print("\n🚀 End-to-End Query Processing Example")
    print("=" * 50)
    
    query = "What genes are associated with Alzheimer's disease?"
    print(f"Complex Query: {query}")
    
    result = await handle_plan_and_execute({
        "query": query,
        "max_results": 20,
        "k": 3,
        "confidence_threshold": 0.7
    })
    
    if "error" not in result:
        print("\n✅ End-to-end processing completed!")
        
        # Extract key information from the structured response
        results = result.get("results", [])
        entities = result.get("entities", {})
        query_type = result.get("query_type", "Unknown")
        
        print(f"  • Query Type: {query_type}")
        print(f"  • Entities Found: {len(entities)}")
        print(f"  • Relationships: {len(results)}")
        
        # Show sample entities
        if entities:
            print(f"\n🧬 Sample Entities:")
            for entity, data in list(entities.items())[:3]:
                entity_id = data.get("id", "Unknown") 
                entity_type = data.get("type", "Unknown")
                print(f"    • {entity}: {entity_id} ({entity_type})")
        
        # Show sample relationships
        if results:
            print(f"\n🔗 Sample Relationships:")
            for i, rel in enumerate(results[:3]):
                subject = rel.get("subject", "Unknown")
                predicate = rel.get("predicate", "unknown_relation")
                obj = rel.get("object", "Unknown")
                clean_predicate = predicate.replace("biolink:", "").replace("_", " ")
                print(f"    {i+1}. {subject} --{clean_predicate}--> {obj}")
    else:
        print(f"❌ Error: {result['error']}")
    
    return result


async def main():
    """Run all MCP tool demonstrations"""
    print("🧬 Agentic BTE MCP Tools Demo")
    print("=" * 70)
    print("This demo shows how to use the MCP tools programmatically")
    print("=" * 70)
    
    # Check environment
    if not os.getenv("AGENTIC_BTE_OPENAI_API_KEY"):
        print("⚠️ Warning: AGENTIC_BTE_OPENAI_API_KEY not set")
        print("Some features may not work without an OpenAI API key")
        print()
    
    try:
        # Step 1: Entity Recognition
        ner_result = await demo_bio_ner()
        
        # Extract entity data for next steps
        entity_data = {}
        if "result_data" in ner_result:
            entity_data = ner_result["result_data"].get("entity_ids", {})
        
        # Step 2: TRAPI Building  
        trapi_result = await demo_trapi_building(entity_data)
        
        # Extract TRAPI query for next step
        trapi_query = None
        if "trapi_query" in trapi_result:
            trapi_query = trapi_result["trapi_query"]
        
        # Step 3: BTE API (only if we have a valid TRAPI query)
        if trapi_query:
            await demo_bte_api(trapi_query)
        
        # Step 4: End-to-end processing
        await demo_end_to_end()
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("  • Try running the MCP server: agentic-bte-mcp")
        print("  • Use with Claude Desktop or other MCP clients")
        print("  • Explore the LangGraph multi-agent examples")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("This might be due to missing dependencies or API keys")


if __name__ == "__main__":
    asyncio.run(main())