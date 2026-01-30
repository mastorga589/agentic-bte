#!/usr/bin/env python3
"""
Comprehensive Validation Tests for Production GoT System

This module provides thorough testing of the production-ready Graph of Thoughts
framework to ensure it works correctly end-to-end with real queries.
"""

import asyncio
import time
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_bte.core.queries.production_got_optimizer import (
    ProductionGoTOptimizer,
    ProductionConfig,
    execute_biomedical_query,
    run_biomedical_query
)
from agentic_bte.core.queries.result_presenter import QueryResult
from agentic_bte.core.queries.mcp_integration import get_mcp_integration

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test configuration
TEST_CONFIG = ProductionConfig(
    show_debug=True,
    show_graphs=False,  # Disable graphs for automated testing
    save_results=False,  # Don't save files during testing
    mcp_timeout=30,      # Shorter timeout for tests
    mcp_max_retries=2,   # Fewer retries for faster tests
    max_iterations=3,    # Fewer iterations for faster tests
    confidence_threshold=0.5,  # Lower threshold for testing
    quality_threshold=0.05     # Lower threshold for testing
)


class TestQueries:
    """Test query collections for different complexity levels"""
    
    SIMPLE_QUERIES = [
        "What genes are associated with diabetes?",
        "How does TP53 interact with other proteins?",
        "What drugs target EGFR?",
        "What pathways involve insulin?",
        "What proteins interact with BRCA1?"
    ]
    
    MODERATE_QUERIES = [
        "How do mutations in CFTR lead to cystic fibrosis symptoms?",
        "What is the relationship between oxidative stress and Alzheimer's disease?",
        "How does insulin signaling affect glucose metabolism?",
        "What are the downstream effects of p53 activation?",
        "How do BRCA1 mutations contribute to breast cancer risk?"
    ]
    
    COMPLEX_QUERIES = [
        "What is the molecular network connecting oxidative stress and neurodegeneration in Parkinson's disease?",
        "Construct a comprehensive analysis of the molecular interactions between inflammation and cancer metastasis",
        "How do epigenetic modifications regulate gene expression in diabetes pathogenesis?",
        "What are the interconnected pathways linking metabolism, immunity, and aging?",
        "Describe the molecular mechanisms connecting circadian rhythms to metabolic disorders"
    ]
    
    EDGE_CASES = [
        "Unknown protein XYZ123",
        "Non-existent disease ABC syndrome",
        "Very short query",
        "",  # Empty query
        "A" * 1000,  # Very long query
    ]


class ProductionGoTTester:
    """Comprehensive tester for production GoT system"""
    
    def __init__(self, config: ProductionConfig = None):
        self.config = config or TEST_CONFIG
        self.results: List[Dict[str, Any]] = []
        self.start_time: float = 0.0
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests and return comprehensive results"""
        print("🧪 STARTING COMPREHENSIVE VALIDATION TESTS")
        print("=" * 80)
        
        self.start_time = time.time()
        
        # Test categories
        test_results = {
            'simple_queries': self._test_simple_queries(),
            'moderate_queries': self._test_moderate_queries(), 
            'complex_queries': self._test_complex_queries(),
            'edge_cases': self._test_edge_cases(),
            'performance_tests': self._test_performance(),
            'integration_tests': self._test_integration(),
            'error_handling': self._test_error_handling()
        }
        
        # Generate summary
        summary = self._generate_test_summary(test_results)
        
        # Print results
        self._print_test_results(test_results, summary)
        
        return {
            'test_results': test_results,
            'summary': summary,
            'total_execution_time': time.time() - self.start_time
        }
    
    def _test_simple_queries(self) -> Dict[str, Any]:
        """Test simple biomedical queries"""
        print("\n📋 Testing Simple Queries")
        print("-" * 40)
        
        results = []
        successful = 0
        
        for i, query in enumerate(TestQueries.SIMPLE_QUERIES):
            print(f"\n{i+1}. Testing: {query}")
            try:
                result, presentation = run_biomedical_query(query, self.config)
                
                test_result = {
                    'query': query,
                    'success': result.success,
                    'execution_time': result.total_execution_time,
                    'entities_found': len(result.entities_found),
                    'results_count': result.total_results,
                    'quality_score': result.quality_score,
                    'error': result.error_message if not result.success else None
                }
                
                results.append(test_result)
                
                if result.success:
                    successful += 1
                    print(f"   ✅ SUCCESS - {result.total_results} results, quality: {result.quality_score:.3f}")
                else:
                    print(f"   ❌ FAILED - {result.error_message}")
                
            except Exception as e:
                print(f"   💥 EXCEPTION - {str(e)}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'execution_time': 0,
                    'entities_found': 0,
                    'results_count': 0,
                    'quality_score': 0.0
                })
        
        success_rate = successful / len(TestQueries.SIMPLE_QUERIES)
        print(f"\n📊 Simple Queries Summary: {successful}/{len(TestQueries.SIMPLE_QUERIES)} successful ({success_rate:.1%})")
        
        return {
            'results': results,
            'successful': successful,
            'total': len(TestQueries.SIMPLE_QUERIES),
            'success_rate': success_rate,
            'avg_execution_time': sum(r.get('execution_time', 0) for r in results) / len(results),
            'avg_quality_score': sum(r.get('quality_score', 0) for r in results if r.get('quality_score', 0) > 0) / max(1, len([r for r in results if r.get('quality_score', 0) > 0]))
        }
    
    def _test_moderate_queries(self) -> Dict[str, Any]:
        """Test moderate complexity queries"""
        print("\n📋 Testing Moderate Complexity Queries")
        print("-" * 40)
        
        results = []
        successful = 0
        
        for i, query in enumerate(TestQueries.MODERATE_QUERIES):
            print(f"\n{i+1}. Testing: {query[:60]}...")
            try:
                result, presentation = run_biomedical_query(query, self.config)
                
                test_result = {
                    'query': query,
                    'success': result.success,
                    'execution_time': result.total_execution_time,
                    'entities_found': len(result.entities_found),
                    'results_count': result.total_results,
                    'quality_score': result.quality_score,
                    'got_volume': result.got_metrics.get('volume', 0),
                    'got_latency': result.got_metrics.get('latency', 0),
                    'error': result.error_message if not result.success else None
                }
                
                results.append(test_result)
                
                if result.success:
                    successful += 1
                    print(f"   ✅ SUCCESS - {result.total_results} results, quality: {result.quality_score:.3f}")
                    print(f"      GoT metrics - Volume: {result.got_metrics.get('volume', 0)}, Latency: {result.got_metrics.get('latency', 0)}")
                else:
                    print(f"   ❌ FAILED - {result.error_message}")
                
            except Exception as e:
                print(f"   💥 EXCEPTION - {str(e)}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'execution_time': 0,
                    'entities_found': 0,
                    'results_count': 0,
                    'quality_score': 0.0,
                    'got_volume': 0,
                    'got_latency': 0
                })
        
        success_rate = successful / len(TestQueries.MODERATE_QUERIES)
        print(f"\n📊 Moderate Queries Summary: {successful}/{len(TestQueries.MODERATE_QUERIES)} successful ({success_rate:.1%})")
        
        return {
            'results': results,
            'successful': successful,
            'total': len(TestQueries.MODERATE_QUERIES),
            'success_rate': success_rate,
            'avg_execution_time': sum(r.get('execution_time', 0) for r in results) / len(results),
            'avg_quality_score': sum(r.get('quality_score', 0) for r in results if r.get('quality_score', 0) > 0) / max(1, len([r for r in results if r.get('quality_score', 0) > 0])),
            'avg_got_volume': sum(r.get('got_volume', 0) for r in results) / len(results),
            'avg_got_latency': sum(r.get('got_latency', 0) for r in results) / len(results)
        }
    
    def _test_complex_queries(self) -> Dict[str, Any]:
        """Test complex research-level queries"""
        print("\n📋 Testing Complex Research Queries")
        print("-" * 40)
        
        results = []
        successful = 0
        
        for i, query in enumerate(TestQueries.COMPLEX_QUERIES):
            print(f"\n{i+1}. Testing: {query[:60]}...")
            try:
                result, presentation = run_biomedical_query(query, self.config)
                
                test_result = {
                    'query': query,
                    'success': result.success,
                    'execution_time': result.total_execution_time,
                    'entities_found': len(result.entities_found),
                    'results_count': result.total_results,
                    'quality_score': result.quality_score,
                    'got_metrics': result.got_metrics,
                    'execution_steps': len(result.execution_steps),
                    'error': result.error_message if not result.success else None
                }
                
                results.append(test_result)
                
                if result.success:
                    successful += 1
                    print(f"   ✅ SUCCESS - {result.total_results} results, quality: {result.quality_score:.3f}")
                    print(f"      Execution steps: {len(result.execution_steps)}, Entities: {len(result.entities_found)}")
                    if result.got_metrics:
                        volume = result.got_metrics.get('volume', 0)
                        latency = result.got_metrics.get('latency', 0)
                        print(f"      GoT metrics - Volume: {volume}, Latency: {latency}")
                else:
                    print(f"   ❌ FAILED - {result.error_message}")
                
            except Exception as e:
                print(f"   💥 EXCEPTION - {str(e)}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'execution_time': 0,
                    'entities_found': 0,
                    'results_count': 0,
                    'quality_score': 0.0,
                    'got_metrics': {},
                    'execution_steps': 0
                })
        
        success_rate = successful / len(TestQueries.COMPLEX_QUERIES)
        print(f"\n📊 Complex Queries Summary: {successful}/{len(TestQueries.COMPLEX_QUERIES)} successful ({success_rate:.1%})")
        
        return {
            'results': results,
            'successful': successful,
            'total': len(TestQueries.COMPLEX_QUERIES),
            'success_rate': success_rate,
            'avg_execution_time': sum(r.get('execution_time', 0) for r in results) / len(results),
            'avg_quality_score': sum(r.get('quality_score', 0) for r in results if r.get('quality_score', 0) > 0) / max(1, len([r for r in results if r.get('quality_score', 0) > 0])),
            'avg_execution_steps': sum(r.get('execution_steps', 0) for r in results) / len(results)
        }
    
    def _test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error conditions"""
        print("\n📋 Testing Edge Cases and Error Conditions")
        print("-" * 40)
        
        results = []
        handled_gracefully = 0
        
        for i, query in enumerate(TestQueries.EDGE_CASES):
            display_query = query if len(query) < 50 else query[:50] + "..."
            print(f"\n{i+1}. Testing edge case: '{display_query}'")
            try:
                result, presentation = run_biomedical_query(query, self.config)
                
                # Edge cases should either succeed or fail gracefully
                graceful = not result.success and result.error_message is not None
                
                test_result = {
                    'query': query,
                    'success': result.success,
                    'handled_gracefully': graceful,
                    'execution_time': result.total_execution_time,
                    'error': result.error_message
                }
                
                results.append(test_result)
                
                if result.success:
                    print(f"   ✅ UNEXPECTED SUCCESS - {result.total_results} results")
                elif graceful:
                    handled_gracefully += 1
                    print(f"   ✅ HANDLED GRACEFULLY - {result.error_message}")
                else:
                    print(f"   ⚠️  NOT HANDLED GRACEFULLY")
                
            except Exception as e:
                print(f"   ❌ UNHANDLED EXCEPTION - {str(e)}")
                results.append({
                    'query': query,
                    'success': False,
                    'handled_gracefully': False,
                    'error': str(e),
                    'execution_time': 0
                })
        
        graceful_rate = handled_gracefully / len(TestQueries.EDGE_CASES)
        print(f"\n📊 Edge Cases Summary: {handled_gracefully}/{len(TestQueries.EDGE_CASES)} handled gracefully ({graceful_rate:.1%})")
        
        return {
            'results': results,
            'handled_gracefully': handled_gracefully,
            'total': len(TestQueries.EDGE_CASES),
            'graceful_rate': graceful_rate
        }
    
    def _test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics"""
        print("\n📋 Testing Performance Characteristics")
        print("-" * 40)
        
        # Test with a representative query multiple times
        test_query = "What genes are associated with diabetes?"
        num_runs = 3
        
        execution_times = []
        memory_usage = []
        
        print(f"Running '{test_query}' {num_runs} times...")
        
        for i in range(num_runs):
            print(f"\n  Run {i+1}/{num_runs}")
            try:
                start_time = time.time()
                result, _ = run_biomedical_query(test_query, self.config)
                execution_time = time.time() - start_time
                
                execution_times.append(execution_time)
                print(f"    Execution time: {execution_time:.3f}s")
                print(f"    Results: {result.total_results}, Quality: {result.quality_score:.3f}")
                
            except Exception as e:
                print(f"    ❌ Failed: {str(e)}")
        
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        min_time = min(execution_times) if execution_times else 0
        max_time = max(execution_times) if execution_times else 0
        
        print(f"\n📊 Performance Summary:")
        print(f"    Average execution time: {avg_time:.3f}s")
        print(f"    Min execution time: {min_time:.3f}s")
        print(f"    Max execution time: {max_time:.3f}s")
        
        return {
            'test_query': test_query,
            'num_runs': num_runs,
            'successful_runs': len(execution_times),
            'execution_times': execution_times,
            'avg_execution_time': avg_time,
            'min_execution_time': min_time,
            'max_execution_time': max_time
        }
    
    def _test_integration(self) -> Dict[str, Any]:
        """Test MCP integration components"""
        print("\n📋 Testing MCP Integration")
        print("-" * 40)
        
        integration_results = {
            'mcp_tools_available': True,
            'bio_ner_working': False,
            'build_trapi_working': False,
            'call_bte_api_working': False
        }
        
        # Test MCP integration availability
        try:
            mcp_integration = get_mcp_integration()
            print("   ✅ MCP integration initialized")
        except Exception as e:
            print(f"   ❌ MCP integration failed: {str(e)}")
            integration_results['mcp_tools_available'] = False
            return integration_results
        
        # Test individual MCP tools
        test_query = "diabetes genes"
        
        # Test bio_ner
        try:
            result = asyncio.run(mcp_integration.call_mcp_tool("bio_ner", query=test_query))
            if 'entities' in result:
                print("   ✅ bio_ner tool working")
                integration_results['bio_ner_working'] = True
            else:
                print("   ⚠️  bio_ner tool returned unexpected format")
        except Exception as e:
            print(f"   ❌ bio_ner tool failed: {str(e)}")
        
        # Test build_trapi_query
        try:
            result = asyncio.run(mcp_integration.call_mcp_tool(
                "build_trapi_query", 
                query=test_query, 
                entity_data={},
                failed_trapis=[]
            ))
            if 'query' in result:
                print("   ✅ build_trapi_query tool working")
                integration_results['build_trapi_working'] = True
            else:
                print("   ⚠️  build_trapi_query tool returned unexpected format")
        except Exception as e:
            print(f"   ❌ build_trapi_query tool failed: {str(e)}")
        
        # Test call_bte_api
        try:
            sample_query = {
                "message": {
                    "query_graph": {
                        "nodes": {"n0": {"categories": ["biolink:Gene"]}},
                        "edges": {}
                    }
                }
            }
            result = asyncio.run(mcp_integration.call_mcp_tool(
                "call_bte_api",
                json_query=sample_query,
                k=5,
                maxresults=10
            ))
            if 'results' in result:
                print("   ✅ call_bte_api tool working")
                integration_results['call_bte_api_working'] = True
            else:
                print("   ⚠️  call_bte_api tool returned unexpected format")
        except Exception as e:
            print(f"   ❌ call_bte_api tool failed: {str(e)}")
        
        working_tools = sum([
            integration_results['bio_ner_working'],
            integration_results['build_trapi_working'],
            integration_results['call_bte_api_working']
        ])
        
        print(f"\n📊 Integration Summary: {working_tools}/3 MCP tools working")
        
        return integration_results
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery"""
        print("\n📋 Testing Error Handling and Recovery")
        print("-" * 40)
        
        # Test with intentionally problematic configurations
        error_configs = [
            ProductionConfig(mcp_timeout=1),  # Very short timeout
            ProductionConfig(max_iterations=0),  # Zero iterations
            ProductionConfig(confidence_threshold=1.1),  # Invalid threshold
        ]
        
        error_handling_results = []
        
        for i, config in enumerate(error_configs):
            print(f"\n  Testing error config {i+1}")
            try:
                result, _ = run_biomedical_query("What genes are associated with diabetes?", config)
                
                error_handling_results.append({
                    'config_index': i,
                    'handled_gracefully': not result.success and result.error_message is not None,
                    'error_message': result.error_message
                })
                
                if not result.success and result.error_message:
                    print(f"    ✅ Error handled gracefully: {result.error_message}")
                else:
                    print(f"    ⚠️  Unexpected success or poor error handling")
                    
            except Exception as e:
                print(f"    ❌ Unhandled exception: {str(e)}")
                error_handling_results.append({
                    'config_index': i,
                    'handled_gracefully': False,
                    'error_message': str(e)
                })
        
        graceful_count = sum(1 for r in error_handling_results if r['handled_gracefully'])
        graceful_rate = graceful_count / len(error_configs)
        
        print(f"\n📊 Error Handling Summary: {graceful_count}/{len(error_configs)} errors handled gracefully ({graceful_rate:.1%})")
        
        return {
            'results': error_handling_results,
            'graceful_count': graceful_count,
            'total_configs': len(error_configs),
            'graceful_rate': graceful_rate
        }
    
    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall test summary"""
        
        # Calculate overall success rates
        total_tests = 0
        total_successful = 0
        
        for category in ['simple_queries', 'moderate_queries', 'complex_queries']:
            if category in test_results:
                total_tests += test_results[category]['total']
                total_successful += test_results[category]['successful']
        
        overall_success_rate = total_successful / total_tests if total_tests > 0 else 0
        
        # Calculate average execution time
        all_times = []
        for category in ['simple_queries', 'moderate_queries', 'complex_queries']:
            if category in test_results:
                all_times.extend([r.get('execution_time', 0) for r in test_results[category]['results']])
        
        avg_execution_time = sum(all_times) / len(all_times) if all_times else 0
        
        return {
            'total_tests': total_tests,
            'total_successful': total_successful,
            'overall_success_rate': overall_success_rate,
            'avg_execution_time': avg_execution_time,
            'integration_status': test_results.get('integration_tests', {}),
            'performance_status': test_results.get('performance_tests', {}),
            'error_handling_status': test_results.get('error_handling', {})
        }
    
    def _print_test_results(self, test_results: Dict[str, Any], summary: Dict[str, Any]):
        """Print comprehensive test results"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 80)
        
        # Overall summary
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Successful: {summary['total_successful']}")
        print(f"   Success Rate: {summary['overall_success_rate']:.1%}")
        print(f"   Average Execution Time: {summary['avg_execution_time']:.3f}s")
        
        # Category breakdown
        print(f"\n📋 CATEGORY BREAKDOWN:")
        for category, data in test_results.items():
            if isinstance(data, dict) and 'success_rate' in data:
                print(f"   {category.replace('_', ' ').title()}: {data['success_rate']:.1%} ({data['successful']}/{data['total']})")
        
        # Integration status
        integration = summary.get('integration_status', {})
        if integration:
            working_tools = sum([
                integration.get('bio_ner_working', False),
                integration.get('build_trapi_working', False),
                integration.get('call_bte_api_working', False)
            ])
            print(f"   MCP Integration: {working_tools}/3 tools working")
        
        # Performance metrics
        performance = summary.get('performance_status', {})
        if performance:
            print(f"   Performance: Avg {performance.get('avg_execution_time', 0):.3f}s per query")
        
        # Error handling
        error_handling = summary.get('error_handling_status', {})
        if error_handling:
            print(f"   Error Handling: {error_handling.get('graceful_rate', 0):.1%} graceful")
        
        # Final assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        if summary['overall_success_rate'] >= 0.8:
            print("   ✅ EXCELLENT - System is production ready!")
        elif summary['overall_success_rate'] >= 0.6:
            print("   ✅ GOOD - System is mostly functional with minor issues")
        elif summary['overall_success_rate'] >= 0.4:
            print("   ⚠️  FAIR - System has significant issues that need addressing")
        else:
            print("   ❌ POOR - System has major problems and is not ready for production")
        
        print("\n" + "=" * 80)


def main():
    """Main test execution function"""
    print("🚀 Starting Production GoT System Validation")
    print("This will run comprehensive tests to validate the system is production ready.")
    print()
    
    tester = ProductionGoTTester()
    results = tester.run_all_tests()
    
    print(f"\n⏱️  Total test execution time: {results['total_execution_time']:.2f}s")
    
    # Return appropriate exit code
    success_rate = results['summary']['overall_success_rate']
    if success_rate >= 0.8:
        return 0  # Success
    elif success_rate >= 0.6:
        return 0  # Acceptable
    else:
        return 1  # Failure


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        sys.exit(1)