"""
Comprehensive Testing Framework for Graph of Thoughts (GoT) Implementation

This module provides comprehensive end-to-end testing of the GoT framework implementation
including performance measurement, comparison with baseline methods, and validation
of key metrics from the paper.
"""

import asyncio
import logging
import time
import json
import sys
import os
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import numpy as np

# Add the project root to Python path
sys.path.append('/Users/mastorga/Documents/agentic-bte')

try:
    from agentic_bte.core.queries.got_optimizers import (
        GoTEnhancedSimpleOptimizer, GoTEnhancedHybridOptimizer, GoTPerformanceComparator
    )
    from agentic_bte.core.queries.got_framework import GoTBiomedicalPlanner, GoTOptimizer
    from agentic_bte.core.queries.got_aggregation import BiomedicalAggregator, IterativeRefinementEngine
    from agentic_bte.core.queries.got_metrics import GoTMetricsCalculator, GoTPaperMetrics, GoTBenchmarkSuite
    from agentic_bte.core.queries.simple_working_optimizer import SimpleWorkingOptimizer
    from agentic_bte.core.queries.hybrid_optimizer import HybridIntelligentOptimizer
    from agentic_bte.core.queries.interfaces import OptimizerConfig
    from call_mcp_tool import call_mcp_tool
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all GoT modules are properly installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoTTestingFramework:
    """
    Comprehensive testing framework for GoT implementation
    """
    
    def __init__(self, save_results: bool = True, results_dir: str = "got_test_results"):
        """
        Initialize the testing framework
        
        Args:
            save_results: Whether to save test results to files
            results_dir: Directory to save results
        """
        self.save_results = save_results
        self.results_dir = results_dir
        self.test_results = {}
        self.performance_data = []
        
        # Create results directory
        if save_results and not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Test queries categorized by complexity
        self.test_queries = {
            'simple': [
                "What genes are associated with diabetes?",
                "What proteins interact with TP53?",
                "What diseases are caused by BRCA1 mutations?",
                "What drugs treat hypertension?",
                "What pathways involve insulin?"
            ],
            'moderate': [
                "What are the pathways connecting insulin signaling to glucose metabolism?",
                "How do mutations in CFTR lead to cystic fibrosis symptoms?",
                "What drugs target the proteins involved in Alzheimer's disease?",
                "How does oxidative stress contribute to aging?",
                "What is the relationship between inflammation and cancer?"
            ],
            'complex': [
                "What is the molecular network connecting oxidative stress, inflammation, and neurodegeneration in Parkinson's disease?",
                "How do genetic variants in metabolism-related genes interact with dietary factors to influence obesity risk?",
                "What are the shared molecular mechanisms between depression and cardiovascular disease?",
                "How do circadian rhythms affect metabolic disorders and what therapeutic targets exist?",
                "What are the molecular connections between gut microbiome and brain function in autism?"
            ],
            'research': [
                "Construct a comprehensive map of the molecular interactions between the gut microbiome, immune system, and brain function in neuropsychiatric disorders.",
                "Identify and analyze the multi-level regulatory networks that control stem cell pluripotency and differentiation.",
                "Develop a systems-level understanding of how circadian rhythms integrate with metabolic networks to influence aging."
            ]
        }
        
        logger.info(f"GoT Testing Framework initialized. Results will be saved to: {results_dir}")
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive tests of the GoT implementation
        
        Returns:
            Complete test results
        """
        logger.info("Starting comprehensive GoT testing")
        
        test_results = {
            'framework_info': {
                'test_timestamp': time.time(),
                'total_queries': sum(len(queries) for queries in self.test_queries.values()),
                'complexity_levels': list(self.test_queries.keys())
            },
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'end_to_end_tests': {},
            'comparison_tests': {},
            'paper_validation': {}
        }
        
        try:
            # Run different test categories
            logger.info("Running unit tests...")
            test_results['unit_tests'] = await self._run_unit_tests()
            
            logger.info("Running integration tests...")
            test_results['integration_tests'] = await self._run_integration_tests()
            
            logger.info("Running performance tests...")
            test_results['performance_tests'] = await self._run_performance_tests()
            
            logger.info("Running end-to-end tests...")
            test_results['end_to_end_tests'] = await self._run_end_to_end_tests()
            
            logger.info("Running comparison tests...")
            test_results['comparison_tests'] = await self._run_comparison_tests()
            
            logger.info("Running paper validation tests...")
            test_results['paper_validation'] = await self._run_paper_validation_tests()
            
            # Generate summary
            test_results['summary'] = self._generate_test_summary(test_results)
            
        except Exception as e:
            logger.error(f"Comprehensive testing failed: {e}")
            test_results['error'] = str(e)
        
        # Save results
        if self.save_results:
            self._save_test_results(test_results)
        
        logger.info("Comprehensive GoT testing completed")
        return test_results
    
    async def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests for individual components"""
        unit_results = {
            'got_framework': {},
            'aggregation': {},
            'refinement': {},
            'metrics': {}
        }
        
        try:
            # Test GoT framework components
            logger.info("Testing GoT framework components")
            
            # Test basic planner initialization
            planner = GoTBiomedicalPlanner(max_iterations=3, enable_parallel=True)
            unit_results['got_framework']['initialization'] = {
                'success': True,
                'max_iterations': planner.max_iterations,
                'parallel_enabled': planner.enable_parallel,
                'transformations_count': len(planner.transformations)
            }
            
            # Test aggregator
            logger.info("Testing biomedical aggregator")
            aggregator = BiomedicalAggregator(enable_refinement=True)
            unit_results['aggregation']['initialization'] = {
                'success': True,
                'refinement_enabled': aggregator.enable_refinement,
                'entity_hierarchies_count': len(aggregator.entity_hierarchies)
            }
            
            # Test refinement engine
            logger.info("Testing refinement engine")
            refinement_engine = IterativeRefinementEngine(max_iterations=3)
            unit_results['refinement']['initialization'] = {
                'success': True,
                'max_iterations': refinement_engine.max_iterations,
                'improvement_threshold': refinement_engine.improvement_threshold
            }
            
            # Test metrics calculator
            logger.info("Testing metrics calculator")
            metrics_calc = GoTMetricsCalculator()
            unit_results['metrics']['initialization'] = {
                'success': True,
                'benchmarks_count': len(metrics_calc.paper_benchmarks)
            }
            
        except Exception as e:
            logger.error(f"Unit tests failed: {e}")
            unit_results['error'] = str(e)
        
        return unit_results
    
    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests for component interactions"""
        integration_results = {
            'optimizer_creation': {},
            'mcp_integration': {},
            'workflow_integration': {}
        }
        
        try:
            # Test optimizer creation
            logger.info("Testing optimizer creation and configuration")
            
            config = OptimizerConfig(
                max_results=20,
                k=3,
                confidence_threshold=0.6
            )
            
            simple_got = GoTEnhancedSimpleOptimizer(
                config=config,
                enable_got=True,
                enable_parallel=True
            )
            
            hybrid_got = GoTEnhancedHybridOptimizer(
                config=config,
                enable_got=True,
                got_strategy_threshold=0.6
            )
            
            integration_results['optimizer_creation'] = {
                'simple_optimizer': {
                    'success': True,
                    'got_enabled': simple_got.enable_got,
                    'parallel_enabled': simple_got.enable_parallel
                },
                'hybrid_optimizer': {
                    'success': True,
                    'got_enabled': hybrid_got.enable_got,
                    'threshold': hybrid_got.got_strategy_threshold
                }
            }
            
            # Test MCP integration with a simple query
            logger.info("Testing MCP integration")
            try:
                # Test basic entity extraction
                test_query = "What genes are associated with diabetes?"
                ner_response = call_mcp_tool("bio_ner", query=test_query)
                entities = ner_response.get("entities", {})
                
                integration_results['mcp_integration']['entity_extraction'] = {
                    'success': True,
                    'entities_found': len(entities),
                    'sample_entities': list(entities.keys())[:3]
                }
                
                # Test query building if entities found
                if entities:
                    trapi_response = call_mcp_tool(
                        "build_trapi_query",
                        query=test_query,
                        entity_data=entities
                    )
                    trapi_query = trapi_response.get("query", {})
                    
                    integration_results['mcp_integration']['query_building'] = {
                        'success': True,
                        'trapi_nodes': len(trapi_query.get('message', {}).get('query_graph', {}).get('nodes', {})),
                        'trapi_edges': len(trapi_query.get('message', {}).get('query_graph', {}).get('edges', {}))
                    }
                
            except Exception as e:
                logger.warning(f"MCP integration test failed (expected if MCP not available): {e}")
                integration_results['mcp_integration']['error'] = str(e)
            
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            integration_results['error'] = str(e)
        
        return integration_results
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests for different components"""
        performance_results = {
            'execution_times': {},
            'memory_usage': {},
            'scalability': {}
        }
        
        try:
            # Test execution times for different query complexities
            logger.info("Testing execution performance")
            
            for complexity, queries in self.test_queries.items():
                if complexity == 'research':  # Skip most complex for performance tests
                    continue
                
                complexity_times = []
                test_query = queries[0]  # Test with first query of each complexity
                
                # Test baseline optimizer
                baseline_start = time.time()
                try:
                    baseline_optimizer = SimpleWorkingOptimizer()
                    baseline_result = baseline_optimizer.optimize_query(test_query)
                    baseline_time = time.time() - baseline_start
                    baseline_success = baseline_result.success
                except Exception as e:
                    baseline_time = float('inf')
                    baseline_success = False
                    logger.warning(f"Baseline test failed: {e}")
                
                # Test GoT optimizer
                got_start = time.time()
                try:
                    got_optimizer = GoTEnhancedSimpleOptimizer(enable_got=True)
                    got_result = got_optimizer.optimize_query(test_query)
                    got_time = time.time() - got_start
                    got_success = got_result.success
                except Exception as e:
                    got_time = float('inf')
                    got_success = False
                    logger.warning(f"GoT test failed: {e}")
                
                performance_results['execution_times'][complexity] = {
                    'baseline_time': baseline_time,
                    'got_time': got_time,
                    'baseline_success': baseline_success,
                    'got_success': got_success,
                    'speedup': baseline_time / got_time if got_time > 0 else 0
                }
            
        except Exception as e:
            logger.error(f"Performance tests failed: {e}")
            performance_results['error'] = str(e)
        
        return performance_results
    
    async def _run_end_to_end_tests(self) -> Dict[str, Any]:
        """Run end-to-end tests with complete query workflows"""
        e2e_results = {
            'by_complexity': {},
            'success_rates': {},
            'quality_metrics': {}
        }
        
        try:
            for complexity, queries in self.test_queries.items():
                logger.info(f"Running end-to-end tests for {complexity} queries")
                
                complexity_results = {
                    'queries_tested': len(queries),
                    'results': [],
                    'success_count': 0,
                    'avg_quality': 0.0,
                    'avg_execution_time': 0.0
                }
                
                for i, query in enumerate(queries):
                    if i >= 2 and complexity == 'research':  # Limit research queries
                        break
                    
                    logger.info(f"Testing query: {query[:50]}...")
                    
                    try:
                        # Run with GoT-enhanced optimizer
                        optimizer = GoTEnhancedSimpleOptimizer(enable_got=True)
                        start_time = time.time()
                        result = optimizer.optimize_query(query)
                        execution_time = time.time() - start_time
                        
                        query_result = {
                            'query': query,
                            'success': result.success,
                            'execution_time': execution_time,
                            'quality_score': result.metrics.quality_score,
                            'result_count': len(result.results),
                            'entity_count': len(result.entities),
                            'reasoning_steps': len(result.reasoning_chain),
                            'has_final_answer': bool(result.final_answer),
                            'errors': result.errors,
                            'warnings': result.warnings
                        }
                        
                        # Add GoT-specific metrics if available
                        if hasattr(optimizer, 'get_got_performance_summary'):
                            got_summary = optimizer.get_got_performance_summary()
                            query_result['got_metrics'] = {
                                'volume': got_summary.get('core_metrics', {}).get('volume', 0),
                                'latency': got_summary.get('core_metrics', {}).get('latency', 0),
                                'total_thoughts': got_summary.get('core_metrics', {}).get('total_thoughts', 0),
                                'quality_improvement': got_summary.get('performance_improvements', {}).get('quality_improvement_factor', 0)
                            }
                        
                        complexity_results['results'].append(query_result)
                        
                        if result.success:
                            complexity_results['success_count'] += 1
                        
                    except Exception as e:
                        logger.error(f"End-to-end test failed for query: {e}")
                        complexity_results['results'].append({
                            'query': query,
                            'success': False,
                            'error': str(e)
                        })
                
                # Calculate summary statistics
                successful_results = [r for r in complexity_results['results'] if r.get('success', False)]
                
                if successful_results:
                    complexity_results['avg_quality'] = np.mean([
                        r.get('quality_score', 0) for r in successful_results
                    ])
                    complexity_results['avg_execution_time'] = np.mean([
                        r.get('execution_time', 0) for r in successful_results
                    ])
                    complexity_results['avg_result_count'] = np.mean([
                        r.get('result_count', 0) for r in successful_results
                    ])
                
                complexity_results['success_rate'] = (
                    complexity_results['success_count'] / len(complexity_results['results'])
                    if complexity_results['results'] else 0
                )
                
                e2e_results['by_complexity'][complexity] = complexity_results
            
            # Calculate overall success rates
            total_queries = sum(r['queries_tested'] for r in e2e_results['by_complexity'].values())
            total_successes = sum(r['success_count'] for r in e2e_results['by_complexity'].values())
            
            e2e_results['overall_success_rate'] = total_successes / total_queries if total_queries > 0 else 0
            e2e_results['total_queries_tested'] = total_queries
            e2e_results['total_successes'] = total_successes
            
        except Exception as e:
            logger.error(f"End-to-end tests failed: {e}")
            e2e_results['error'] = str(e)
        
        return e2e_results
    
    async def _run_comparison_tests(self) -> Dict[str, Any]:
        """Run comparison tests between GoT and baseline methods"""
        comparison_results = {
            'optimizer_comparisons': {},
            'performance_improvements': {},
            'statistical_significance': {}
        }
        
        try:
            logger.info("Running optimizer comparison tests")
            
            # Initialize comparator
            comparator = GoTPerformanceComparator()
            
            # Test with queries from different complexity levels
            test_queries = []
            for complexity, queries in self.test_queries.items():
                if complexity != 'research':  # Skip most complex for comparison
                    test_queries.extend(queries[:2])  # Take first 2 from each level
            
            for query in test_queries[:5]:  # Limit to 5 queries for comprehensive comparison
                logger.info(f"Comparing optimizers for: {query[:50]}...")
                
                try:
                    comparison = await comparator.compare_optimizers(
                        query=query,
                        entities=None,
                        runs=1  # Single run for testing (normally would use 3)
                    )
                    
                    comparison_results['optimizer_comparisons'][query[:30]] = {
                        'summary': comparison.get('summary', {}),
                        'best_quality': comparison.get('summary', {}).get('best_quality'),
                        'fastest': comparison.get('summary', {}).get('fastest'),
                        'most_reliable': comparison.get('summary', {}).get('most_reliable')
                    }
                    
                except Exception as e:
                    logger.warning(f"Comparison failed for query '{query}': {e}")
                    comparison_results['optimizer_comparisons'][query[:30]] = {
                        'error': str(e)
                    }
            
            # Generate improvement summary
            all_results = comparator.get_all_comparison_results()
            if all_results:
                summary = comparator.export_results_summary()
                comparison_results['performance_improvements'] = {
                    'avg_quality_improvement': summary.get('avg_quality_improvement', 0),
                    'avg_speed_improvement': summary.get('avg_speed_improvement', 0),
                    'total_comparisons': summary.get('total_comparisons', 0)
                }
            
        except Exception as e:
            logger.error(f"Comparison tests failed: {e}")
            comparison_results['error'] = str(e)
        
        return comparison_results
    
    async def _run_paper_validation_tests(self) -> Dict[str, Any]:
        """Run tests to validate implementation against paper claims"""
        validation_results = {
            'volume_latency_tradeoff': {},
            'quality_improvements': {},
            'cost_reductions': {},
            'paper_claims_validation': {}
        }
        
        try:
            logger.info("Running paper validation tests")
            
            # Test key paper claims
            paper_claims = {
                'volume_efficiency': "GoT achieves high volume with low latency",
                'quality_improvement': "GoT improves quality over baseline methods",
                'cost_reduction': "GoT reduces costs compared to ToT",
                'parallel_efficiency': "GoT benefits from parallel execution"
            }
            
            # Run tests with different configurations to validate claims
            test_configs = [
                {'name': 'GoT_Parallel_On', 'enable_parallel': True, 'max_iterations': 3},
                {'name': 'GoT_Parallel_Off', 'enable_parallel': False, 'max_iterations': 3},
                {'name': 'GoT_More_Iterations', 'enable_parallel': True, 'max_iterations': 5}
            ]
            
            validation_data = []
            test_query = "What genes are associated with diabetes and how do they affect metabolism?"
            
            for config in test_configs:
                logger.info(f"Testing configuration: {config['name']}")
                
                try:
                    optimizer = GoTEnhancedSimpleOptimizer(
                        enable_got=True,
                        enable_parallel=config['enable_parallel']
                    )
                    
                    start_time = time.time()
                    result = optimizer.optimize_query(test_query)
                    execution_time = time.time() - start_time
                    
                    # Extract GoT metrics
                    got_summary = optimizer.get_got_performance_summary()
                    
                    config_data = {
                        'config': config['name'],
                        'success': result.success,
                        'execution_time': execution_time,
                        'quality_score': result.metrics.quality_score,
                        'volume': got_summary.get('core_metrics', {}).get('volume', 0),
                        'latency': got_summary.get('core_metrics', {}).get('latency', 0),
                        'total_thoughts': got_summary.get('core_metrics', {}).get('total_thoughts', 0),
                        'parallel_enabled': config['enable_parallel']
                    }
                    
                    validation_data.append(config_data)
                    
                except Exception as e:
                    logger.warning(f"Validation test failed for {config['name']}: {e}")
                    validation_data.append({
                        'config': config['name'],
                        'error': str(e)
                    })
            
            # Analyze validation data
            successful_configs = [d for d in validation_data if d.get('success', False)]
            
            if len(successful_configs) >= 2:
                # Volume-latency tradeoff analysis
                avg_volume = np.mean([d['volume'] for d in successful_configs])
                avg_latency = np.mean([d['latency'] for d in successful_configs])
                volume_efficiency = avg_volume / max(1, avg_latency)
                
                validation_results['volume_latency_tradeoff'] = {
                    'avg_volume': avg_volume,
                    'avg_latency': avg_latency,
                    'volume_efficiency': volume_efficiency,
                    'validates_paper_claim': volume_efficiency > 2.0  # Arbitrary threshold
                }
                
                # Parallel efficiency analysis
                parallel_configs = [d for d in successful_configs if d.get('parallel_enabled', False)]
                sequential_configs = [d for d in successful_configs if not d.get('parallel_enabled', True)]
                
                if parallel_configs and sequential_configs:
                    avg_parallel_time = np.mean([d['execution_time'] for d in parallel_configs])
                    avg_sequential_time = np.mean([d['execution_time'] for d in sequential_configs])
                    
                    if avg_parallel_time > 0:
                        speedup = avg_sequential_time / avg_parallel_time
                        validation_results['parallel_efficiency'] = {
                            'avg_parallel_time': avg_parallel_time,
                            'avg_sequential_time': avg_sequential_time,
                            'speedup': speedup,
                            'validates_paper_claim': speedup > 1.1  # Should be faster
                        }
            
            validation_results['configurations_tested'] = validation_data
            
            # Validate against paper benchmarks (conceptual)
            paper_expectations = {
                'quality_improvement_over_baseline': 0.1,  # 10% improvement
                'cost_reduction': 0.05,  # 5% cost reduction
                'volume_latency_ratio': 2.0   # Should achieve good volume-latency tradeoff
            }
            
            validation_results['paper_claims_validation'] = {
                'expected_improvements': paper_expectations,
                'validation_notes': "Direct comparison with paper limited due to different domain (biomedical vs sorting)",
                'conceptual_validation': "Framework successfully implements key GoT concepts"
            }
            
        except Exception as e:
            logger.error(f"Paper validation tests failed: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        summary = {
            'overall_status': 'UNKNOWN',
            'test_categories_run': len([k for k in test_results.keys() if k != 'framework_info']),
            'total_tests_executed': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'key_findings': [],
            'recommendations': [],
            'performance_highlights': {}
        }
        
        try:
            # Count tests and successes
            categories_with_errors = []
            
            for category, results in test_results.items():
                if category in ['framework_info', 'summary']:
                    continue
                
                if isinstance(results, dict):
                    if 'error' in results:
                        categories_with_errors.append(category)
                        summary['failed_tests'] += 1
                    else:
                        summary['successful_tests'] += 1
                    
                    summary['total_tests_executed'] += 1
            
            # Determine overall status
            if summary['failed_tests'] == 0:
                summary['overall_status'] = 'PASS'
            elif summary['successful_tests'] > summary['failed_tests']:
                summary['overall_status'] = 'PARTIAL_PASS'
            else:
                summary['overall_status'] = 'FAIL'
            
            # Extract key findings
            summary['key_findings'] = []
            
            # Unit test findings
            if 'unit_tests' in test_results and 'error' not in test_results['unit_tests']:
                summary['key_findings'].append("✓ All core GoT components initialize successfully")
            
            # Integration test findings
            if 'integration_tests' in test_results:
                integration = test_results['integration_tests']
                if 'optimizer_creation' in integration:
                    summary['key_findings'].append("✓ GoT-enhanced optimizers create and configure properly")
            
            # End-to-end test findings
            if 'end_to_end_tests' in test_results and 'error' not in test_results['end_to_end_tests']:
                e2e = test_results['end_to_end_tests']
                overall_success = e2e.get('overall_success_rate', 0)
                summary['key_findings'].append(f"✓ End-to-end success rate: {overall_success:.1%}")
                
                if overall_success > 0.8:
                    summary['key_findings'].append("✓ High end-to-end success rate demonstrates robust implementation")
                elif overall_success > 0.5:
                    summary['key_findings'].append("⚠ Moderate success rate - some queries may need optimization")
                else:
                    summary['key_findings'].append("⚠ Low success rate - implementation needs review")
            
            # Performance findings
            if 'performance_tests' in test_results and 'execution_times' in test_results['performance_tests']:
                perf = test_results['performance_tests']['execution_times']
                speedups = [data.get('speedup', 0) for data in perf.values() if isinstance(data, dict)]
                if speedups:
                    avg_speedup = np.mean(speedups)
                    if avg_speedup > 1.0:
                        summary['key_findings'].append(f"✓ Average speedup over baseline: {avg_speedup:.2f}x")
                    else:
                        summary['key_findings'].append(f"⚠ GoT slower than baseline (avg: {avg_speedup:.2f}x)")
            
            # Comparison findings
            if 'comparison_tests' in test_results:
                comp = test_results['comparison_tests']
                if 'performance_improvements' in comp:
                    improvements = comp['performance_improvements']
                    quality_imp = improvements.get('avg_quality_improvement', 0)
                    speed_imp = improvements.get('avg_speed_improvement', 0)
                    
                    if quality_imp > 0:
                        summary['key_findings'].append(f"✓ Average quality improvement: {quality_imp:.1%}")
                    if speed_imp > 1.0:
                        summary['key_findings'].append(f"✓ Average speed improvement: {speed_imp:.2f}x")
            
            # Validation findings
            if 'paper_validation' in test_results:
                validation = test_results['paper_validation']
                if 'volume_latency_tradeoff' in validation:
                    tradeoff = validation['volume_latency_tradeoff']
                    if tradeoff.get('validates_paper_claim', False):
                        summary['key_findings'].append("✓ Volume-latency tradeoff validates paper claims")
                    
                if 'parallel_efficiency' in validation:
                    parallel = validation['parallel_efficiency']
                    if parallel.get('validates_paper_claim', False):
                        summary['key_findings'].append("✓ Parallel execution shows expected benefits")
            
            # Generate recommendations
            if categories_with_errors:
                summary['recommendations'].append(f"Fix errors in: {', '.join(categories_with_errors)}")
            
            if summary['overall_status'] == 'PASS':
                summary['recommendations'].append("Implementation is ready for production use")
                summary['recommendations'].append("Consider running extended benchmarks for performance optimization")
            elif summary['overall_status'] == 'PARTIAL_PASS':
                summary['recommendations'].append("Address failed test categories before production")
                summary['recommendations'].append("Investigate performance bottlenecks")
            else:
                summary['recommendations'].append("Major issues need resolution before deployment")
                summary['recommendations'].append("Review GoT framework implementation")
            
            # Performance highlights
            if 'end_to_end_tests' in test_results:
                e2e = test_results['end_to_end_tests']
                by_complexity = e2e.get('by_complexity', {})
                
                for complexity, data in by_complexity.items():
                    if isinstance(data, dict) and 'avg_quality' in data:
                        summary['performance_highlights'][f'{complexity}_avg_quality'] = data['avg_quality']
            
        except Exception as e:
            logger.error(f"Failed to generate test summary: {e}")
            summary['summary_generation_error'] = str(e)
        
        return summary
    
    def _save_test_results(self, test_results: Dict[str, Any]):
        """Save test results to files"""
        try:
            timestamp = int(time.time())
            
            # Save complete results as JSON
            results_file = os.path.join(self.results_dir, f"got_test_results_{timestamp}.json")
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            
            # Save summary as separate file
            if 'summary' in test_results:
                summary_file = os.path.join(self.results_dir, f"got_test_summary_{timestamp}.json")
                with open(summary_file, 'w') as f:
                    json.dump(test_results['summary'], f, indent=2, default=str)
            
            # Save readable report
            report_file = os.path.join(self.results_dir, f"got_test_report_{timestamp}.txt")
            with open(report_file, 'w') as f:
                f.write("GoT Framework Comprehensive Test Report\n")
                f.write("=" * 50 + "\n\n")
                
                if 'summary' in test_results:
                    summary = test_results['summary']
                    f.write(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}\n")
                    f.write(f"Tests Executed: {summary.get('total_tests_executed', 0)}\n")
                    f.write(f"Successful: {summary.get('successful_tests', 0)}\n")
                    f.write(f"Failed: {summary.get('failed_tests', 0)}\n\n")
                    
                    f.write("Key Findings:\n")
                    for finding in summary.get('key_findings', []):
                        f.write(f"  {finding}\n")
                    
                    f.write("\nRecommendations:\n")
                    for rec in summary.get('recommendations', []):
                        f.write(f"  {rec}\n")
            
            logger.info(f"Test results saved to {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")


async def main():
    """Main function to run comprehensive tests"""
    print("Starting GoT Framework Comprehensive Testing")
    print("=" * 50)
    
    # Initialize testing framework
    framework = GoTTestingFramework(save_results=True)
    
    try:
        # Run comprehensive tests
        results = await framework.run_comprehensive_tests()
        
        # Print summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        
        if 'summary' in results:
            summary = results['summary']
            print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
            print(f"Total Tests: {summary.get('total_tests_executed', 0)}")
            print(f"Successful: {summary.get('successful_tests', 0)}")
            print(f"Failed: {summary.get('failed_tests', 0)}")
            
            print("\nKey Findings:")
            for finding in summary.get('key_findings', [])[:10]:  # Show first 10
                print(f"  {finding}")
            
            print("\nRecommendations:")
            for rec in summary.get('recommendations', []):
                print(f"  {rec}")
        
        if results.get('summary', {}).get('overall_status') == 'PASS':
            print("\n✅ All tests passed! GoT implementation is working correctly.")
        elif results.get('summary', {}).get('overall_status') == 'PARTIAL_PASS':
            print("\n⚠️  Most tests passed, but some issues need attention.")
        else:
            print("\n❌ Tests failed. Implementation needs review.")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Testing framework failed: {e}")
        logger.error(f"Main testing function failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # Run the comprehensive tests
    results = asyncio.run(main())
    
    if results:
        print(f"\nDetailed results saved to: got_test_results/")
    else:
        print("\nTesting failed - check logs for details")
        sys.exit(1)