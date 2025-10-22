#!/usr/bin/env python3
"""
Quick test script for local BTE functionality with agentic-bte
"""

import asyncio
from agentic_bte.core.queries.production_got_optimizer import execute_biomedical_query

async def test_local_bte():
    print("🧬 TESTING LOCAL BTE WITH AGENTIC-BTE")
    print("=" * 50)
    
    # Simple test query
    query = "What genes are related to aspirin?"
    
    try:
        print(f"🎯 Query: {query}")
        print("⚡ Executing with local BTE...")
        
        result, presentation = await execute_biomedical_query(query)
        
        print(f"✅ Success: {result.success}")
        print(f"📊 Results found: {result.total_results}")
        print(f"🔍 Entities: {len(result.entities_found)}")
        
        # Show first part of answer
        if result.final_answer:
            print("\n📋 ANSWER PREVIEW:")
            print(result.final_answer[:300] + "..." if len(result.final_answer) > 300 else result.final_answer)
        
        print("\n🎉 LOCAL BTE IS WORKING WITH AGENTIC-BTE!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 TROUBLESHOOTING:")
        print("   1. Ensure local BTE is running on localhost:3000")
        print("   2. Check BTE logs for errors")
        print("   3. Try restarting the local BTE instance")

if __name__ == "__main__":
    asyncio.run(test_local_bte())
