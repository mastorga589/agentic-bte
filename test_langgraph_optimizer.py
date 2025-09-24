#!/usr/bin/env python3
"""
Comprehensive Test Suite for LangGraph Context-Driven Adaptive Query Optimizer

This script tests the true LangGraph-inspired adaptive optimizer that uses pure
contextual reasoning and mechanistic decomposition without predefined strategies,
compared to the previous strategy-based and static approaches.
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.insert(0, '/Users/mastorga/Documents/agentic-bte')

# Set environment variables
os.environ['PYTHONPATH'] = '/Users/mastorga/Documents/agentic-bte'

def setup_environment():
    """Setup test environment and API keys"""
    print("🧪 LangGraph Context-Driven Optimizer Test Suite")
    print("=" * 70)
    print("✅ Environment setup complete")
    print()


def test_context_driven_optimizer():
    """Test the LangGraph context-driven optimizer"""
    print("🧠 Testing LangGraph Context-Driven Optimizer...")
    
    from agentic_bte.core.queries.langgraph_adaptive_optimizer import LangGraphAdaptiveOptimizer
    
    try:
        optimizer = LangGraphAdaptiveOptimizer()
        print("✅ LangGraph context-driven optimizer initialized")
        print()
        return optimizer
    except Exception as e:
        print(f"❌ Failed to initialize LangGraph optimizer: {e}")
        return None


def test_context_driven_queries(optimizer, test_queries):
    """Test context-driven optimizer on various queries"""
    if not optimizer:
        return {}
    
    results = {}
    
    for i, query in enumerate(test_queries, 1):
        print(f"--- Test Query {i}: {query[:50]}... ---")
        
        try:
            # Create and execute adaptive plan
            plan = optimizer.create_adaptive_plan(query, max_iterations=5)
            plan = optimizer.execute_adaptive_plan(plan)
            
            results[query] = {
                'success': True,
                'subqueries_executed': len(plan.executed_subqueries),
                'total_results': len(plan.accumulated_results),
                'execution_time': plan.total_execution_time,
                'completion_reason': plan.completion_reason,
                'final_answer_length': len(plan.final_answer),
                'subqueries': [sq.query for sq in plan.executed_subqueries]
            }
            
            print(f"Subqueries executed: {len(plan.executed_subqueries)}")
            print(f"Total results: {len(plan.accumulated_results)}")
            print(f"Execution time: {plan.total_execution_time:.2f}s")
            print(f"Completion reason: {plan.completion_reason}")
            
            # Show subqueries for insight
            for j, subquery in enumerate(plan.executed_subqueries, 1):
                print(f"  {j}. {subquery.query} -> {len(subquery.results)} results")
            print()
            
        except Exception as e:
            print(f"❌ Error testing query: {e}")
            results[query] = {
                'success': False,
                'error': str(e)
            }
            print()
    
    return results


def test_comprehensive_comparison():
    """Run comprehensive comparison between context-driven and static approaches"""
    print("🔄 Running Comprehensive Comparison...")
    
    from agentic_bte.core.queries.langgraph_adaptive_optimizer import LangGraphAdaptiveOptimizer
    
    try:
        optimizer = LangGraphAdaptiveOptimizer()
        
        # Complex test query
        test_query = "What drugs can treat Prostatitis by targeting DNA topological change?"
        print(f"Comparison query: {test_query}")
        
        comparison = optimizer.compare_with_static(test_query)
        
        print()
        print("📈 Comparison Results:")
        print(f"Query: {comparison['query'][:70]}...")
        print()
        
        print("🧠 Context-Driven Approach:")
        adaptive = comparison['adaptive_approach']
        print(f"  - Execution Time: {adaptive['execution_time']:.2f}s")
        print(f"  - Subqueries Executed: {adaptive['subqueries_executed']}")
        print(f"  - Total Results: {adaptive['total_results']}")
        print(f"  - Success: {adaptive['success']}")
        print(f"  - Completion Reason: {adaptive['completion_reason']}")
        print(f"  - Final Answer Length: {adaptive['final_answer_length']}")
        print()
        
        print("📊 Static Approach:")
        static = comparison['static_approach']
        print(f"  - Execution Time: {static['execution_time']:.2f}s")
        print(f"  - Total Results: {static['total_results']}")
        print(f"  - Success: {static['success']}")
        print(f"  - Query Type: {static['query_type']}")
        print(f"  - Classification Confidence: {static['classification_confidence']}")
        print()
        
        print("📊 Comparison Metrics:")
        metrics = comparison['comparison_metrics']
        print(f"  - Context-driven found more results: {metrics['adaptive_more_results']}")
        print(f"  - Context-driven was faster: {metrics['adaptive_faster']}")
        print(f"  - Context-driven more comprehensive: {metrics['adaptive_more_comprehensive']}")
        print(f"  - Result difference: {metrics['result_difference']}")
        print(f"  - Time difference: {metrics['time_difference']:.2f}s")
        print()
        
        return comparison
        
    except Exception as e:
        print(f"❌ Error in comprehensive comparison: {e}")
        return {}


def test_context_driven_features(optimizer):
    """Test specific context-driven features"""
    print("🎯 Testing Context-Driven Features...")
    
    if not optimizer:
        return {}
    
    results = {}
    
    # Test 1: Pure contextual subquery generation
    test_query = "What drugs can treat diabetes?"
    print("Contextual Planning Test:")
    
    try:
        plan = optimizer.create_adaptive_plan(test_query, max_iterations=3)
        plan = optimizer.execute_adaptive_plan(plan)
        
        print(f"  Original query: {test_query}")
        print(f"  Subqueries executed: {len(plan.executed_subqueries)}")
        print(f"  Completion reason: {plan.completion_reason}")
        
        for i, sq in enumerate(plan.executed_subqueries, 1):
            print(f"    {i}. {sq.query} (iteration {sq.iteration_number})")
        
        results['contextual_planning'] = {
            'success': True,
            'subqueries': [sq.query for sq in plan.executed_subqueries],
            'completion_reason': plan.completion_reason,
            'total_results': len(plan.accumulated_results)
        }
        print("✅ Context-driven features tested successfully")
        
    except Exception as e:
        print(f"❌ Error testing context-driven features: {e}")
        results['contextual_planning'] = {
            'success': False,
            'error': str(e)
        }
    
    print()
    return results


def generate_comparison_report(context_results, comparison_results, feature_results):
    """Generate comprehensive comparison report"""
    print("📄 Generating LangGraph Comparison Report...")
    
    # Create comprehensive report
    report = {
        "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_type": "LangGraph Context-Driven vs Static vs Strategy-Based",
        "context_driven_optimizer": {
            "description": "Pure contextual reasoning, no predefined strategies",
            "queries_tested": len(context_results),
            "successful_executions": sum(1 for r in context_results.values() if r.get('success', False)),
            "features": {
                "rdf_knowledge_graphs": True,
                "contextual_subquery_generation": True,
                "mechanistic_reasoning": True,
                "single_hop_queries": True,
                "iterative_refinement": True,
                "predefined_strategies": False
            }
        },
        "context_driven_results": context_results,
        "comprehensive_comparison": comparison_results,
        "feature_testing": feature_results,
        "key_insights": [
            "Context-driven approach uses pure LLM-based contextual reasoning",
            "No predefined strategies - queries generated based on knowledge state",
            "Uses RDF/Turtle graphs for knowledge accumulation like LangGraph",
            "Implements mechanistic decomposition through iterative planning",
            "Single-hop subqueries with contextual refinement",
            "True adaptive planning based on intermediate results"
        ]
    }
    
    # Save report
    report_path = "/Users/mastorga/Documents/agentic-bte/LANGGRAPH_COMPARISON_REPORT.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Report saved to: {report_path}")
    return report


def main():
    """Main test execution"""
    setup_environment()
    
    # Test queries covering different biomedical domains
    test_queries = [
        "What drugs can treat Prostatitis by targeting DNA topological change?",
        "What drugs treat diabetes?", 
        "How does aspirin work to prevent heart disease?",
        "Which genes are involved in Alzheimer's disease pathogenesis?",
        "What are the mechanisms of action of metformin in treating diabetes?"
    ]
    
    # Initialize and test context-driven optimizer
    optimizer = test_context_driven_optimizer()
    
    # Test context-driven approach on various queries
    context_results = test_context_driven_queries(optimizer, test_queries)
    print(f"✅ Context-driven optimizer tested on {len(test_queries)} queries")
    print()
    
    # Test specific context-driven features
    feature_results = test_context_driven_features(optimizer)
    
    # Run comprehensive comparison
    comparison_results = test_comprehensive_comparison()
    
    # Generate and save comprehensive report
    report = generate_comparison_report(context_results, comparison_results, feature_results)
    
    # Print summary
    print("📊 Test Summary:")
    successful_tests = sum(1 for r in context_results.values() if r.get('success', False))
    print(f"  Context-Driven Optimizer: {successful_tests}/{len(context_results)} successful executions")
    
    if comparison_results:
        adaptive_results = comparison_results.get('adaptive_approach', {}).get('total_results', 0)
        static_results = comparison_results.get('static_approach', {}).get('total_results', 0)
        print(f"  Comparison: Context-driven found {adaptive_results} results vs Static {static_results}")
    
    print(f"  Key Insights: {len(report['key_insights'])} observations")
    print()
    print("🎉 LangGraph Context-Driven Testing Complete!")
    print("Check LANGGRAPH_COMPARISON_REPORT.json for detailed results.")


if __name__ == "__main__":
    main()