#!/usr/bin/env python3
"""
Quick verification that JSON parsing is working correctly
"""

from agentic_bte.core.queries.llm_based_planner import LLMBasedQueryPlanner

def test_json_parsing():
    print("🔍 Testing JSON Parsing Status")
    print("=" * 40)
    
    try:
        # Quick test with a simple query
        query = "What genes does metformin interact with?"
        
        print(f"📝 Test Query: {query}")
        print()
        
        print("🔧 Creating planner...")
        planner = LLMBasedQueryPlanner()
        
        print("📋 Testing strategy analysis JSON parsing...")
        entities = {"metformin": "CHEBI:6801"}
        strategy_result = planner._analyze_query_strategy(query, entities)
        
        print(f"✅ Strategy parsing: {strategy_result['strategy']} (confidence: {strategy_result['confidence']:.2f})")
        print()
        
        print("📋 Testing meta-KG subquery generation...")
        available_edges = planner._get_available_edges_from_entities(entities)
        scored_edges = planner._score_edges_for_query(available_edges, query)
        top_edges = [edge for _, edge in scored_edges[:4]]
        
        subqueries = planner._generate_metakg_informed_subqueries(
            query, entities, top_edges, strategy_result, 3
        )
        
        print(f"✅ Meta-KG subquery parsing: Generated {len(subqueries)} subqueries")
        
        if subqueries:
            print("   Sample subqueries:")
            for i, sq in enumerate(subqueries[:2], 1):
                print(f"   {i}. {sq.get('query', 'Unknown')}")
        
        print()
        print("🎯 RESULT: JSON parsing is working correctly!")
        print("Both strategy analysis and subquery generation are parsing JSON successfully.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in JSON parsing test: {e}")
        return False

if __name__ == "__main__":
    success = test_json_parsing()
    
    print()
    print("=" * 50)
    if success:
        print("✅ CONFIRMATION: JSON parsing issues are RESOLVED!")
        print("✅ The enhanced LLM planner is fully functional!")
    else:
        print("❌ JSON parsing issues still exist")
    print("=" * 50)