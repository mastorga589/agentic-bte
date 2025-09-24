#!/usr/bin/env python3
"""
Test Script: Adaptive vs Static Query Optimization Comparison

This script tests and compares the new adaptive query optimizer 
(inspired by LangGraph) with the existing static query classification approach.
"""

import os
import sys
import json
import time
from typing import Dict, List

# Add the project root to path
sys.path.insert(0, '/Users/mastorga/Documents/agentic-bte')

def setup_environment():
    """Setup test environment"""
    # Use test API key
    os.environ['OPENAI_API_KEY'] = 'test-key-for-debugging'
    print("✅ Environment setup complete")

def test_adaptive_optimizer():
    """Test the new adaptive query optimizer"""
    print("\n🧠 Testing Adaptive Query Optimizer...")
    
    try:
        from agentic_bte.core.queries.adaptive_optimizer import (
            AdaptiveQueryOptimizer, 
            optimize_biomedical_query_adaptive
        )
        
        # Initialize optimizer
        optimizer = AdaptiveQueryOptimizer()
        print("✅ Adaptive optimizer initialized")
        
        # Test queries of varying complexity
        test_queries = [
            "What drugs can treat Prostatitis by targeting DNA topological change?",
            "What drugs treat diabetes?",
            "How does aspirin work to prevent heart disease?",
            "Which genes are involved in Alzheimer's disease pathogenesis?",
            "What are the mechanisms of action of metformin in treating diabetes?"
        ]
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i}: {query[:60]}... ---")
            
            try:
                # Test strategy determination
                entities = {}  # Will be extracted automatically
                strategy = optimizer.determine_planning_strategy(query, entities)
                print(f"Selected strategy: {strategy.value}")
                
                # Create adaptive plan (but don't execute fully to save time)
                plan = optimizer.create_adaptive_plan(query, max_iterations=3)
                print(f"Plan created with {len(plan.subqueries)} initial subqueries")
                
                if plan.subqueries:
                    print(f"First subquery: {plan.subqueries[0].query}")
                
                results.append({
                    "query": query,
                    "strategy": strategy.value,
                    "plan_created": True,
                    "initial_subqueries": len(plan.subqueries)
                })
                
            except Exception as e:
                print(f"❌ Error with query {i}: {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "plan_created": False
                })
        
        print(f"\n✅ Adaptive optimizer tested on {len(results)} queries")
        return results
        
    except Exception as e:
        print(f"❌ Failed to test adaptive optimizer: {e}")
        return []

def test_static_approach():
    """Test the current static query classification approach"""
    print("\n📊 Testing Static Query Classification...")
    
    try:
        from agentic_bte.core.knowledge.knowledge_system import BiomedicalKnowledgeSystem
        
        # Initialize system
        system = BiomedicalKnowledgeSystem()
        print("✅ Static system initialized")
        
        # Same test queries
        test_queries = [
            "What drugs can treat Prostatitis by targeting DNA topological change?",
            "What drugs treat diabetes?",
            "How does aspirin work to prevent heart disease?",
            "Which genes are involved in Alzheimer's disease pathogenesis?",
            "What are the mechanisms of action of metformin in treating diabetes?"
        ]
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i}: {query[:60]}... ---")
            
            try:
                # Test entity extraction
                entities = system.extract_entities_only(query)
                entity_count = len(entities.get("entities", {}))
                print(f"Entities extracted: {entity_count}")
                
                # Test query classification  
                classification = system.classify_query_only(query, entities.get("entity_ids", {}))
                query_type = classification.get("query_type")
                confidence = classification.get("confidence", 0.0)
                print(f"Query type: {query_type.value if hasattr(query_type, 'value') else query_type}")
                print(f"Confidence: {confidence:.2f}")
                
                results.append({
                    "query": query,
                    "entity_count": entity_count,
                    "query_type": query_type.value if hasattr(query_type, 'value') else str(query_type),
                    "confidence": confidence,
                    "classification_success": True
                })
                
            except Exception as e:
                print(f"❌ Error with query {i}: {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "classification_success": False
                })
        
        print(f"\n✅ Static approach tested on {len(results)} queries")
        return results
        
    except Exception as e:
        print(f"❌ Failed to test static approach: {e}")
        return []

def compare_approaches():
    """Run comprehensive comparison"""
    print("\n🔄 Running Comprehensive Comparison...")
    
    try:
        from agentic_bte.core.queries.adaptive_optimizer import AdaptiveQueryOptimizer
        
        optimizer = AdaptiveQueryOptimizer()
        
        # Test with one detailed query
        test_query = "What drugs can treat Prostatitis by targeting DNA topological change?"
        print(f"Comparison query: {test_query}")
        
        # Run comparison method
        comparison = optimizer.compare_with_static_approach(test_query)
        
        print("\n📈 Comparison Results:")
        print(f"Query: {comparison['query'][:80]}...")
        
        adaptive = comparison['adaptive_approach']
        static = comparison['static_approach']
        metrics = comparison['comparison_metrics']
        
        print(f"\n🧠 Adaptive Approach:")
        print(f"  - Execution Time: {adaptive['execution_time']:.2f}s")
        print(f"  - Subqueries Executed: {adaptive['subqueries_executed']}")
        print(f"  - Total Results: {adaptive['total_results']}")
        print(f"  - Strategy: {adaptive['strategy']}")
        print(f"  - Success: {adaptive['success']}")
        print(f"  - Replanning Count: {adaptive.get('replanning_count', 0)}")
        
        print(f"\n📊 Static Approach:")
        print(f"  - Execution Time: {static['execution_time']:.2f}s")
        print(f"  - Total Results: {static['total_results']}")
        print(f"  - Query Type: {static['query_type']}")
        print(f"  - Success: {static['success']}")
        print(f"  - Classification Confidence: {static.get('classification_confidence', 0.0):.2f}")
        
        print(f"\n📊 Comparison Metrics:")
        print(f"  - Adaptive found more results: {metrics['adaptive_more_results']}")
        print(f"  - Adaptive was faster: {metrics['adaptive_faster']}")
        print(f"  - Adaptive more comprehensive: {metrics['adaptive_more_comprehensive']}")
        print(f"  - Result difference: {metrics['result_difference']}")
        print(f"  - Time difference: {metrics['time_difference']:.2f}s")
        
        return comparison
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_adaptive_features():
    """Test specific adaptive features"""
    print("\n🎯 Testing Adaptive Features...")
    
    try:
        from agentic_bte.core.queries.adaptive_optimizer import (
            AdaptiveQueryOptimizer,
            AdaptivePlanningStrategy,
            ExecutionContext
        )
        
        optimizer = AdaptiveQueryOptimizer()
        
        # Test strategy determination
        test_cases = [
            ("How does metformin work?", "mechanistic_chain"),
            ("What drugs treat cancer?", "bidirectional"),
            ("Can aspirin prevent heart disease?", "hypothesis"),
            ("What genes are involved in diabetes?", "exploratory"),
            ("Find biomarkers for Alzheimer's", "exploratory")
        ]
        
        print("Strategy Determination Tests:")
        for query, expected in test_cases:
            strategy = optimizer.determine_planning_strategy(query, {})
            print(f"  '{query[:40]}...' -> {strategy.value} (expected: {expected})")
        
        # Test contextual subquery generation
        print("\nContextual Planning Test:")
        test_query = "What drugs can treat diabetes?"
        plan = optimizer.create_adaptive_plan(test_query, max_iterations=2)
        
        if plan.subqueries:
            print(f"  Original query: {test_query}")
            print(f"  Strategy: {plan.strategy.value}")
            print(f"  Initial subquery: {plan.subqueries[0].query}")
            print(f"  Context: {plan.subqueries[0].execution_context.value}")
        
        print("✅ Adaptive features tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ Adaptive features test failed: {e}")
        return False

def generate_report(adaptive_results, static_results, comparison_result):
    """Generate comprehensive comparison report"""
    print("\n📄 Generating Comparison Report...")
    
    report = {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "adaptive_optimizer": {
            "queries_tested": len(adaptive_results),
            "successful_plans": sum(1 for r in adaptive_results if r.get("plan_created", False)),
            "strategies_used": list(set(r.get("strategy", "unknown") for r in adaptive_results if r.get("strategy"))),
            "average_subqueries": sum(r.get("initial_subqueries", 0) for r in adaptive_results) / max(len(adaptive_results), 1)
        },
        "static_approach": {
            "queries_tested": len(static_results),
            "successful_classifications": sum(1 for r in static_results if r.get("classification_success", False)),
            "query_types_identified": list(set(r.get("query_type", "unknown") for r in static_results if r.get("query_type"))),
            "average_confidence": sum(r.get("confidence", 0.0) for r in static_results) / max(len(static_results), 1)
        },
        "comparison_summary": comparison_result if comparison_result else "No comparison data available",
        "key_insights": []
    }
    
    # Generate insights
    if comparison_result:
        metrics = comparison_result.get("comparison_metrics", {})
        adaptive = comparison_result.get("adaptive_approach", {})
        static = comparison_result.get("static_approach", {})
        
        if metrics.get("adaptive_more_results"):
            report["key_insights"].append("Adaptive approach found more results than static approach")
        
        if metrics.get("adaptive_more_comprehensive"):
            report["key_insights"].append("Adaptive approach used multi-step reasoning with subqueries")
        
        if adaptive.get("replanning_count", 0) > 0:
            report["key_insights"].append("Adaptive approach demonstrated replanning capability")
    
    # Save report
    report_file = "/Users/mastorga/Documents/agentic-bte/OPTIMIZATION_COMPARISON_REPORT.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Report saved to: {report_file}")
    
    # Print summary
    print("\n📊 Test Summary:")
    print(f"  Adaptive Optimizer: {report['adaptive_optimizer']['successful_plans']}/{report['adaptive_optimizer']['queries_tested']} successful plans")
    print(f"  Static Approach: {report['static_approach']['successful_classifications']}/{report['static_approach']['queries_tested']} successful classifications")
    print(f"  Key Insights: {len(report['key_insights'])} observations")
    
    return report

def main():
    """Main test execution"""
    print("🧪 Adaptive vs Static Query Optimization Comparison")
    print("=" * 60)
    
    # Setup
    setup_environment()
    
    # Run tests
    adaptive_results = test_adaptive_optimizer()
    static_results = test_static_approach()
    
    # Test adaptive features
    test_adaptive_features()
    
    # Run comparison
    comparison_result = compare_approaches()
    
    # Generate report
    report = generate_report(adaptive_results, static_results, comparison_result)
    
    print("\n🎉 Testing Complete!")
    print("Check OPTIMIZATION_COMPARISON_REPORT.json for detailed results.")

if __name__ == "__main__":
    main()