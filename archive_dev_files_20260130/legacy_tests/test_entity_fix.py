#!/usr/bin/env python3
"""
Test the entity resolution fix
"""

import asyncio
from agentic_bte.core.queries.production_got_optimizer import execute_biomedical_query

async def test_fix():
    print("🧪 TESTING ENTITY RESOLUTION FIX")
    print("=" * 50)
    
    # Simple test query
    query = "What genes are related to aspirin?"
    print(f"🎯 Test Query: {query}")
    
    try:
        result, presentation = await execute_biomedical_query(query)
        
        print(f"📊 Results: Success={result.success}, Total={result.total_results}")
        
        if result.success and result.total_results > 0:
            print("✅ Fix has been applied! Entity names should now be resolved.")
            print(f"📋 Check the final answer for proper entity names (not UMLS IDs)")
        else:
            print("❌ No results returned, cannot test entity name resolution")
            
    except Exception as e:
        print(f"💥 Error: {e}")
        
if __name__ == "__main__":
    asyncio.run(test_fix())