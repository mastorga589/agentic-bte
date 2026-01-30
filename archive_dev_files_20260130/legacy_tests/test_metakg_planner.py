#!/usr/bin/env python3
"""
Meta-KG Aware LLM Planner Test

This script tests the enhanced LLM-based planner that uses meta-KG edges 
to generate subqueries targeting discrete relationships between nodes.
"""

import logging
from agentic_bte.core.queries.llm_based_planner import LLMBasedQueryPlanner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_metakg_aware_planning():
    """Test meta-KG aware planning with discrete relationship subqueries"""
    
    print("🧬 Testing Meta-KG Aware LLM-Based Query Planner")
    print("=" * 60)
    
    # Test query about macular degeneration and antioxidants
    query = "Which drugs can treat Age related macular degeneration by targeting antioxidant activity?"
    
    print(f"📝 Query: {query}")
    print()
    
    try:
        # Initialize planner (this will load the meta-KG)
        print("🔧 Initializing planner with meta-KG loading...")
        planner = LLMBasedQueryPlanner()
        
        print(f"✅ Loaded {len(planner._meta_kg_edges)} meta-KG edges")
        
        # Create advanced plan
        print("📋 Creating meta-KG aware execution plan...")
        plan = planner.create_advanced_plan(
            query=query,
            max_subqueries=6,
            confidence_threshold=0.7
        )
        
        print(f"✅ Plan created with {len(plan.subqueries)} subqueries")
        print()
        
        # Display results
        print("📊 EXECUTION PLAN DETAILS")
        print("-" * 40)
        print(f"Strategy: {plan.strategy.value}")
        print(f"Confidence: {plan.confidence_score:.2f}")
        print(f"Total subqueries: {len(plan.subqueries)}")
        print(f"Execution phases: {len(plan.execution_phases)}")
        print()
        
        print("📋 SUBQUERIES WITH META-KG RELATIONSHIPS:")
        for i, sq in enumerate(plan.subqueries, 1):
            print(f"{i:2d}. {sq.query}")
            
            # Show meta-KG edge info if available
            if sq.suggested_edge:
                edge = sq.suggested_edge
                print(f"     📊 Meta-KG Edge: {edge.subject} --{edge.predicate}--> {edge.object}")
                if edge.api_name:
                    print(f"     🔗 API: {edge.api_name}")
            elif sq.metadata.get('metakg_edge'):
                print(f"     📊 Meta-KG informed: {sq.metadata.get('edge_subject', 'Unknown')} --> {sq.metadata.get('edge_object', 'Unknown')}")
            else:
                print(f"     📋 Strategic subquery (non-meta-KG)")
            
            print(f"     💡 Reasoning: {sq.reasoning}")
            print(f"     🎯 Priority: {sq.priority} | Cost: {sq.estimated_cost:.1f}")
            print()
        
        print("📈 EXECUTION PHASES:")
        for phase_idx, phase in enumerate(plan.execution_phases):
            phase_queries = [sq.query for sq in plan.subqueries if sq.id in phase]
            print(f"  Phase {phase_idx + 1} ({len(phase)} parallel):")
            for query in phase_queries:
                print(f"    • {query[:65]}...")
            print()
        
        # Check for meta-KG utilization
        metakg_subqueries = [sq for sq in plan.subqueries if sq.suggested_edge or sq.metadata.get('metakg_edge')]
        print(f"🎯 Meta-KG Utilization: {len(metakg_subqueries)}/{len(plan.subqueries)} subqueries use meta-KG edges")
        
        if metakg_subqueries:
            print("   ✅ SUCCESS: Subqueries are targeting discrete meta-KG relationships")
            print("   📊 Meta-KG edges being explored:")
            for sq in metakg_subqueries:
                if sq.suggested_edge:
                    print(f"      • {sq.suggested_edge}")
                elif 'edge_predicate' in sq.metadata:
                    print(f"      • {sq.metadata.get('edge_subject')} --{sq.metadata.get('edge_predicate')}--> {sq.metadata.get('edge_object')}")
        else:
            print("   ⚠️  WARNING: No meta-KG informed subqueries generated")
        
        return plan
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_edge_scoring():
    """Test the edge scoring functionality"""
    print("\n🔍 Testing Meta-KG Edge Scoring")
    print("=" * 40)
    
    try:
        planner = LLMBasedQueryPlanner()
        
        # Test query
        query = "What genes does metformin interact with?"
        entities = {"metformin": "CHEBI:6801"}
        
        # Get available edges
        available_edges = planner._get_available_edges_from_entities(entities)
        print(f"📊 Found {len(available_edges)} available edges for entities: {list(entities.keys())}")
        
        # Score edges
        scored_edges = planner._score_edges_for_query(available_edges, query)
        
        print("🏆 Top 5 scored edges:")
        for i, (score, edge) in enumerate(scored_edges[:5], 1):
            print(f"  {i}. Score: {score:4.1f} | {edge}")
            
        return scored_edges[:10]
        
    except Exception as e:
        print(f"❌ Error in edge scoring: {e}")
        return []

def main():
    """Run all tests"""
    print("🧬 Meta-KG Aware LLM Planner Test Suite")
    print("=" * 60)
    print("Testing enhanced planner that uses meta-KG edges to create")
    print("subqueries targeting discrete relationships between nodes.")
    print()
    
    # Test 1: Full planning with meta-KG
    plan = test_metakg_aware_planning()
    
    # Test 2: Edge scoring
    if plan:
        scored_edges = test_edge_scoring()
        
        print("\n🎉 TEST SUMMARY")
        print("=" * 30)
        print(f"✅ Plan created with {len(plan.subqueries)} subqueries")
        
        metakg_count = len([sq for sq in plan.subqueries if sq.suggested_edge or sq.metadata.get('metakg_edge')])
        print(f"✅ {metakg_count} subqueries use meta-KG edges")
        
        if metakg_count > 0:
            print("✅ SUCCESS: Enhanced planner successfully generates discrete relationship subqueries")
            print("   based on meta-KG edges, rather than generic exploratory queries.")
        else:
            print("⚠️  Meta-KG integration may need refinement")
        
        print(f"✅ Edge scoring identified {len(scored_edges)} relevant relationship edges")

if __name__ == "__main__":
    main()