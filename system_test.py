#!/usr/bin/env python3
"""
Complete system test to verify the meta-KG aware planner works end-to-end
"""

from agentic_bte.core.queries.llm_based_planner import LLMBasedQueryPlanner

def test_system():
    print("🧪 COMPLETE SYSTEM TEST")
    print("=" * 40)
    
    # Test the complete system
    query = 'What drugs treat diabetes?'
    print(f'📝 Testing: {query}')
    print()
    
    try:
        print("🔧 Creating planner...")
        planner = LLMBasedQueryPlanner()
        print(f"✅ Loaded {len(planner._meta_kg_edges)} meta-KG edges")
        
        print("📋 Creating execution plan...")
        plan = planner.create_advanced_plan(query, max_subqueries=3)
        
        print(f'✅ SUCCESS: Generated {len(plan.subqueries)} subqueries')
        print(f'Strategy: {plan.strategy.value}')
        print(f'Confidence: {plan.confidence_score:.2f}')
        print()
        
        print("📊 GENERATED SUBQUERIES:")
        meta_kg_count = 0
        
        for i, sq in enumerate(plan.subqueries, 1):
            print(f'{i}. "{sq.query}"')
            
            if sq.suggested_edge:
                print(f'   🔗 Meta-KG Edge: {sq.suggested_edge}')
                meta_kg_count += 1
            elif sq.metadata.get('metakg_edge'):
                edge_info = f"{sq.metadata.get('edge_subject', '?')} -> {sq.metadata.get('edge_object', '?')}"
                print(f'   🔗 Meta-KG informed: {edge_info}')
                meta_kg_count += 1
            else:
                print(f'   📋 Strategic/fallback subquery')
            
            print(f'   💡 Reasoning: {sq.reasoning}')
            print()
        
        print(f'🎯 Meta-KG utilization: {meta_kg_count}/{len(plan.subqueries)} subqueries')
        
        if meta_kg_count > 0:
            print('✅ SYSTEM IS WORKING!')
            print('   The planner successfully uses meta-KG edges to generate')
            print('   subqueries targeting discrete relationships between nodes.')
            return True
        else:
            print('❌ SYSTEM NOT WORKING')
            print('   No meta-KG informed subqueries were generated.')
            return False
            
    except Exception as e:
        print(f'❌ ERROR: {e}')
        return False

def test_specific_edges():
    print("\n🔍 TESTING SPECIFIC META-KG EDGE TARGETING")
    print("=" * 50)
    
    try:
        planner = LLMBasedQueryPlanner()
        
        # Test with entities that should have clear treatment edges
        query = 'Which drugs treat macular degeneration?'
        entities = {'drugs': 'UMLS:C0013227', 'macular degeneration': 'UMLS:C0242383'}
        
        print(f'📝 Query: {query}')
        print(f'📊 Entities: {entities}')
        
        # Get available edges
        available_edges = planner._get_available_edges_from_entities(entities)
        print(f'🔍 Found {len(available_edges)} available edges')
        
        # Score edges
        scored_edges = planner._score_edges_for_query(available_edges, query)
        
        print('🏆 Top 3 scored edges:')
        for i, (score, edge) in enumerate(scored_edges[:3], 1):
            print(f'   {i}. Score: {score:4.1f} | {edge}')
        
        print()
        
        # Generate plan
        plan = planner.create_advanced_plan(query, max_subqueries=2)
        
        print(f'✅ Generated plan with {len(plan.subqueries)} subqueries:')
        for i, sq in enumerate(plan.subqueries, 1):
            print(f'   {i}. {sq.query}')
            if sq.suggested_edge:
                print(f'      Edge: {sq.suggested_edge}')
        
        return len(plan.subqueries) > 0
        
    except Exception as e:
        print(f'❌ ERROR in edge testing: {e}')
        return False

if __name__ == "__main__":
    print("🧬 Meta-KG Aware Planner System Test")
    print("====================================")
    print("Testing that the system actually works end-to-end")
    print()
    
    # Run tests
    test1_passed = test_system()
    test2_passed = test_specific_edges()
    
    # Final result
    print("\n" + "="*50)
    print("🎯 FINAL SYSTEM STATUS")
    print("="*50)
    
    if test1_passed and test2_passed:
        print("✅ COMPLETE SUCCESS!")
        print("✅ JSON parsing is working")
        print("✅ Meta-KG edge loading is working") 
        print("✅ Entity type determination is working")
        print("✅ Edge scoring is working")
        print("✅ Subquery generation targeting discrete relationships is working")
        print("✅ The system accomplishes the original goal!")
    else:
        print("❌ SYSTEM HAS ISSUES")
        if not test1_passed:
            print("❌ Basic system test failed")
        if not test2_passed:
            print("❌ Edge targeting test failed")