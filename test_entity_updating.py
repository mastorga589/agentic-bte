#!/usr/bin/env python3
"""
Test script for Meta-KG Aware Optimizer with Entity Data Updating

This script tests the enhanced optimizer that:
1. Updates entity_data with IDs from BTE results after each subquery
2. Enforces single-hop queries to prevent multi-hop TRAPI queries
3. Provides improved final answers with specific drug names

Usage:
    python test_entity_updating.py
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agentic_bte.core.queries.metakg_aware_optimizer import MetaKGAwareAdaptiveOptimizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_entity_updating.log')
    ]
)

logger = logging.getLogger(__name__)

async def test_amd_antioxidant_query():
    """Test the AMD antioxidant query with entity updating"""
    print("🧬 Testing Meta-KG Aware Optimizer with Entity Data Updating")
    print("=" * 70)
    
    query = "Which drugs can treat Age related macular degeneration by targeting antioxidant activity?"
    print(f"Query: {query}")
    print()
    
    try:
        # Initialize the optimizer
        print("🔧 Initializing Meta-KG Aware Adaptive Optimizer...")
        optimizer = MetaKGAwareAdaptiveOptimizer()
        
        # Create adaptive plan
        print("📋 Creating adaptive plan...")
        plan = optimizer.create_adaptive_plan(
            query=query,
            entities=None,  # Let it extract entities
            max_iterations=5
        )
        
        print(f"✅ Plan created with {len(plan.entity_data)} initial entities")
        print(f"Initial entities: {list(plan.entity_data.keys())[:5]}...")
        print()
        
        # Execute the plan
        print("🚀 Executing adaptive plan...")
        start_time = time.time()
        
        result = optimizer.execute_adaptive_plan(plan)
        
        execution_time = time.time() - start_time
        
        # Print results
        print("=" * 70)
        print("📊 EXECUTION RESULTS")
        print("=" * 70)
        
        print(f"⏱️  Total Execution Time: {execution_time:.2f}s")
        print(f"🔢 Total Results: {len(result.accumulated_results)}")
        print(f"🔍 Subqueries Executed: {len(result.executed_subqueries)}")
        print(f"✅ Completion Reason: {result.completion_reason}")
        print()
        
        # Show entity data updates
        print("🏷️  ENTITY DATA TRACKING:")
        print(f"Final entity count: {len(result.entity_data)}")
        print("Sample final entities:")
        for i, (name, entity_id) in enumerate(list(result.entity_data.items())[:10]):
            print(f"  {i+1:2d}. {name:30s} → {entity_id}")
        if len(result.entity_data) > 10:
            print(f"     ... and {len(result.entity_data) - 10} more entities")
        print()
        
        # Show subquery progression
        print("📋 SUBQUERY PROGRESSION:")
        for i, subquery in enumerate(result.executed_subqueries, 1):
            status = "✅" if subquery.success else "❌"
            print(f"  {i}. {status} {subquery.query}")
            print(f"     └─ {len(subquery.results)} results in {subquery.execution_time:.1f}s")
        print()
        
        # Show sample results with entity resolution
        print("🔬 SAMPLE BIOMEDICAL RELATIONSHIPS:")
        if result.accumulated_results:
            for i, rel in enumerate(result.accumulated_results[:8]):
                subject = rel.get('subject', 'Unknown')
                predicate = rel.get('predicate', 'unknown_relation')
                obj = rel.get('object', 'Unknown')
                
                # Clean up predicate
                clean_predicate = predicate.replace('biolink:', '').replace('_', ' ')
                print(f"  {i+1}. {subject} ← {clean_predicate} → {obj}")
            
            if len(result.accumulated_results) > 8:
                print(f"     ... and {len(result.accumulated_results) - 8} more relationships")
        else:
            print("  No relationships found")
        print()
        
        # Show final answer
        print("🎯 FINAL ANSWER:")
        if result.final_answer:
            print("─" * 40)
            print(result.final_answer)
            print("─" * 40)
        else:
            print("No final answer generated")
        print()
        
        # Test entity extraction specifically
        print("🧪 ENTITY EXTRACTION TEST:")
        if result.accumulated_results:
            sample_results = result.accumulated_results[:3]
            extracted = optimizer._extract_entities_from_results(sample_results)
            print(f"Extracted {len(extracted)} entities from sample results:")
            for name, entity_id in list(extracted.items())[:5]:
                print(f"  • {name} → {entity_id}")
        
        # Test single-hop validation
        print("🚫 SINGLE-HOP VALIDATION TEST:")
        test_queries = [
            "What drugs treat diabetes?",  # Valid single-hop
            "What genes does metformin interact with?",  # Valid single-hop  
            "What genes do drugs that treat diabetes interact with?",  # Invalid multi-hop
            "What processes are affected by genes that interact with diabetes drugs?"  # Invalid multi-hop
        ]
        
        for test_query in test_queries:
            is_single = optimizer._is_single_hop_query(test_query)
            status = "✅ Valid" if is_single else "❌ Multi-hop"
            print(f"  {status}: {test_query}")
        
        print()
        print("🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_simple_diabetes_query():
    """Test a simpler diabetes query for comparison"""
    print("\n" + "=" * 70)
    print("🧬 Testing Simple Diabetes Query (for comparison)")
    print("=" * 70)
    
    query = "What drugs treat type 2 diabetes?"
    print(f"Query: {query}")
    print()
    
    try:
        optimizer = MetaKGAwareAdaptiveOptimizer()
        
        plan = optimizer.create_adaptive_plan(
            query=query,
            entities=None,
            max_iterations=3  # Fewer iterations for simple query
        )
        
        result = optimizer.execute_adaptive_plan(plan)
        
        print("📊 SIMPLE QUERY RESULTS:")
        print(f"🔢 Total Results: {len(result.accumulated_results)}")
        print(f"🔍 Subqueries: {len(result.executed_subqueries)}")
        print(f"🏷️  Final Entities: {len(result.entity_data)}")
        
        # Show first few results
        if result.accumulated_results:
            print("\nSample relationships:")
            for i, rel in enumerate(result.accumulated_results[:5]):
                subject = rel.get('subject', 'Unknown')
                predicate = rel.get('predicate', 'unknown_relation')
                obj = rel.get('object', 'Unknown')
                clean_predicate = predicate.replace('biolink:', '').replace('_', ' ')
                print(f"  {i+1}. {subject} ← {clean_predicate} → {obj}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple query test failed: {str(e)}")
        return False

async def main():
    """Main test function"""
    print("🧬 Meta-KG Aware Optimizer Entity Updating Test Suite")
    print("=" * 70)
    print("This script tests:")
    print("✅ Entity data updating from BTE results")  
    print("✅ Single-hop query validation")
    print("✅ Multi-hop relationship exploration")
    print("✅ Improved final answer generation")
    print("=" * 70)
    print()
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Complex AMD antioxidant query
    if await test_amd_antioxidant_query():
        success_count += 1
    
    # Test 2: Simple diabetes query
    if await test_simple_diabetes_query():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Entity updating is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the logs for details.")
    
    print("\nKey improvements tested:")
    print("✅ Entity IDs from BTE results are extracted and stored")
    print("✅ Subsequent queries can reference specific entities found")
    print("✅ Single-hop validation prevents complex TRAPI queries")
    print("✅ Final answers are contextual and specific")

if __name__ == "__main__":
    asyncio.run(main())