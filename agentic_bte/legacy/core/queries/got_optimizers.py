"""
GoT-Enhanced Optimizers

This module integrates the Graph of Thoughts (GoT) framework with existing
optimizers to provide enhanced biomedical query optimization with graph-based
reasoning, advanced aggregation, and iterative refinement.
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .interfaces import (
    BaseOptimizer, OptimizationStrategy, OptimizationResult, 
    OptimizationMetrics, OptimizerConfig
)
from .simple_working_optimizer import SimpleWorkingOptimizer
from .hybrid_optimizer import HybridIntelligentOptimizer
from .got_framework import GoTBiomedicalPlanner, GoTOptimizer, GoTMetrics
from .got_aggregation import BiomedicalAggregator, IterativeRefinementEngine
from call_mcp_tool import call_mcp_tool

logger = logging.getLogger(__name__)


@dataclass
class GoTPerformanceMetrics:
    """Performance metrics specific to GoT framework based on the paper"""
    # Core GoT metrics from the paper
    volume: int = 0                    # Number of thoughts that could impact final result
    latency: int = 0                   # Number of hops to reach final thought
    total_thoughts: int = 0            # Total thoughts generated
    
    # Quality improvements
    quality_improvement_factor: float = 0.0  # Quality improvement vs baseline
    cost_reduction_factor: float = 0.0       # Cost reduction vs baseline
    
    # Graph characteristics
    graph_complexity: float = 0.0      # Edges/nodes ratio
    aggregation_operations: int = 0     # Number of aggregation operations
    refinement_operations: int = 0      # Number of refinement operations
    
    # Parallel execution metrics
    parallel_batches: int = 0          # Number of parallel execution batches
    parallelization_speedup: float = 0.0  # Speedup from parallelization
    
    # Comparison with baseline (ToT-style) execution
    baseline_execution_time: float = 0.0
    got_execution_time: float = 0.0
    baseline_result_count: int = 0
    got_result_count: int = 0


class GoTEnhancedSimpleOptimizer(SimpleWorkingOptimizer):
    """
    GoT-enhanced version of SimpleWorkingOptimizer with graph-based reasoning
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None, 
                 openai_api_key: Optional[str] = None,
                 enable_got: bool = True,
                 enable_parallel: bool = True):
        """
        Initialize GoT-enhanced simple optimizer
        
        Args:
            config: Optimizer configuration
            openai_api_key: OpenAI API key
            enable_got: Enable GoT framework (if False, falls back to base optimizer)
            enable_parallel: Enable parallel execution in GoT
        """
        super().__init__(config, openai_api_key)
        self.enable_got = enable_got
        self.enable_parallel = enable_parallel
        
        if self.enable_got:
            self.got_optimizer = GoTOptimizer(
                max_iterations=3,
                enable_parallel=enable_parallel
            )
            self.aggregator = BiomedicalAggregator(enable_refinement=True)
            self.refinement_engine = IterativeRefinementEngine()
        
        self.got_metrics = GoTPerformanceMetrics()
        logger.info(f"GoT-enhanced simple optimizer initialized (GoT enabled: {enable_got})")
    
    def get_strategy(self) -> OptimizationStrategy:
        """Return the optimization strategy"""
        return OptimizationStrategy.BASIC_ADAPTIVE
    
    def optimize_query(self, query: str, entities: Optional[Dict[str, str]] = None) -> OptimizationResult:
        """
        Execute a biomedical query using GoT-enhanced optimization
        
        Args:
            query: The biomedical query to execute
            entities: Optional pre-extracted entities
            
        Returns:
            OptimizationResult with GoT enhancements
        """
        if not self.enable_got:
            # Fall back to base optimizer
            return super().optimize_query(query, entities)
        
        # Try to get current event loop, if none exists, create one
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, run the coroutine directly
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self._optimize_query_async(query, entities))
                )
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self._optimize_query_async(query, entities))
    
    async def _optimize_query_async(self, query: str, entities: Optional[Dict[str, str]] = None) -> OptimizationResult:
        """Async implementation of GoT-enhanced query optimization"""
        start_time = time.time()
        
        # For comparison, also run baseline execution
        baseline_start = time.time()
        baseline_result = super().optimize_query(query, entities)
        baseline_time = time.time() - baseline_start
        
        # Run GoT optimization
        got_start = time.time()
        
        try:
            # Execute with GoT framework
            config = {
                'k': self.config.k if self.config else 5,
                'maxresults': self.config.max_results if self.config else 50,
                'query': query
            }
            
            result = await self.got_optimizer.optimize_with_got(
                query=query,
                strategy=self.get_strategy(),
                entities=entities,
                config=config
            )
            
            # Apply iterative refinement if result quality is below threshold
            if result.success and result.metrics.quality_score < 0.7:
                logger.info("Result quality below threshold, applying iterative refinement")
                result = await self.refinement_engine.refine_query_execution(
                    result, {'query': query}
                )
            
            got_time = time.time() - got_start
            
            # Calculate GoT performance metrics
            self._calculate_got_performance_metrics(
                result, baseline_result, baseline_time, got_time
            )
            
            # Enhance result with GoT metrics
            self._enhance_result_with_got_metrics(result)
            
            logger.info(f"GoT-enhanced optimization completed. "
                       f"Quality improvement: {self.got_metrics.quality_improvement_factor:.2f}x, "
                       f"Volume: {self.got_metrics.volume}, Latency: {self.got_metrics.latency}")
            
        except Exception as e:
            logger.error(f"GoT optimization failed, falling back to baseline: {e}")
            # Return baseline result if GoT fails
            result = baseline_result
            result.warnings.append(f"GoT optimization failed: {str(e)}")
        
        self._update_stats(result)
        return result
    
    def _calculate_got_performance_metrics(self, got_result: OptimizationResult,
                                         baseline_result: OptimizationResult,
                                         baseline_time: float,
                                         got_time: float):
        """Calculate performance metrics comparing GoT to baseline"""
        # Get GoT metrics from optimizer
        got_optimizer_metrics = self.got_optimizer.get_metrics()
        
        # Core GoT metrics
        self.got_metrics.volume = got_optimizer_metrics.volume
        self.got_metrics.latency = got_optimizer_metrics.latency
        self.got_metrics.total_thoughts = got_optimizer_metrics.total_thoughts
        self.got_metrics.graph_complexity = got_optimizer_metrics.graph_complexity
        self.got_metrics.aggregation_operations = got_optimizer_metrics.aggregation_count
        self.got_metrics.parallel_batches = got_optimizer_metrics.parallel_executions
        
        # Performance comparisons
        self.got_metrics.baseline_execution_time = baseline_time
        self.got_metrics.got_execution_time = got_time
        self.got_metrics.baseline_result_count = len(baseline_result.results)
        self.got_metrics.got_result_count = len(got_result.results)
        
        # Quality improvement factor
        baseline_quality = baseline_result.metrics.quality_score
        got_quality = got_result.metrics.quality_score
        
        if baseline_quality > 0:
            self.got_metrics.quality_improvement_factor = got_quality / baseline_quality
        else:
            self.got_metrics.quality_improvement_factor = 1.0 if got_quality > 0 else 0.0
        
        # Cost reduction factor (based on result count efficiency)
        if self.got_metrics.baseline_result_count > 0:
            baseline_efficiency = baseline_quality / self.got_metrics.baseline_result_count
            got_efficiency = got_quality / max(1, self.got_metrics.got_result_count)
            self.got_metrics.cost_reduction_factor = got_efficiency / baseline_efficiency
        
        # Parallelization speedup
        if baseline_time > 0:
            self.got_metrics.parallelization_speedup = baseline_time / got_time
    
    def _enhance_result_with_got_metrics(self, result: OptimizationResult):
        """Enhance optimization result with GoT-specific metrics"""
        # Add GoT metrics to the result's reasoning chain
        got_summary = self.got_optimizer.get_execution_summary()
        result.reasoning_chain.append(f"GoT Framework Summary: {got_summary}")
        
        # Update optimization metrics with GoT data
        result.metrics.concurrent_batches = self.got_metrics.parallel_batches
        result.metrics.subqueries_executed = self.got_metrics.total_thoughts
        
        # Add GoT-specific performance data to metadata
        if not hasattr(result, 'got_metrics'):
            result.got_metrics = self.got_metrics
    
    def get_got_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive GoT performance summary"""
        return {
            "got_enabled": self.enable_got,
            "core_metrics": {
                "volume": self.got_metrics.volume,
                "latency": self.got_metrics.latency,
                "total_thoughts": self.got_metrics.total_thoughts,
                "graph_complexity": self.got_metrics.graph_complexity
            },
            "performance_improvements": {
                "quality_improvement_factor": self.got_metrics.quality_improvement_factor,
                "cost_reduction_factor": self.got_metrics.cost_reduction_factor,
                "parallelization_speedup": self.got_metrics.parallelization_speedup
            },
            "execution_comparison": {
                "baseline_time": self.got_metrics.baseline_execution_time,
                "got_time": self.got_metrics.got_execution_time,
                "baseline_results": self.got_metrics.baseline_result_count,
                "got_results": self.got_metrics.got_result_count
            },
            "operations": {
                "aggregation_operations": self.got_metrics.aggregation_operations,
                "refinement_operations": self.got_metrics.refinement_operations,
                "parallel_batches": self.got_metrics.parallel_batches
            }
        }


class GoTEnhancedHybridOptimizer(HybridIntelligentOptimizer):
    """
    GoT-enhanced version of HybridIntelligentOptimizer with advanced graph-based reasoning
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None,
                 openai_api_key: Optional[str] = None,
                 enable_got: bool = True,
                 got_strategy_threshold: float = 0.6):
        """
        Initialize GoT-enhanced hybrid optimizer
        
        Args:
            config: Optimizer configuration
            openai_api_key: OpenAI API key
            enable_got: Enable GoT framework
            got_strategy_threshold: Quality threshold for switching to GoT strategy
        """
        super().__init__(config, openai_api_key)
        self.enable_got = enable_got
        self.got_strategy_threshold = got_strategy_threshold
        
        if self.enable_got:
            # Enhanced GoT optimizer with more iterations for complex queries
            self.got_optimizer = GoTOptimizer(
                max_iterations=5,
                enable_parallel=True
            )
            self.aggregator = BiomedicalAggregator(
                enable_refinement=True,
                max_refinement_iterations=4
            )
            self.refinement_engine = IterativeRefinementEngine(
                max_iterations=4,
                improvement_threshold=0.05
            )
        
        # Track GoT usage per query complexity
        self.complexity_got_usage = {
            'simple': 0,
            'moderate': 0,
            'complex': 0,
            'research': 0
        }
        
        logger.info(f"GoT-enhanced hybrid optimizer initialized "
                   f"(GoT enabled: {enable_got}, threshold: {got_strategy_threshold})")
    
    def get_strategy(self) -> OptimizationStrategy:
        """Return the optimization strategy"""
        return OptimizationStrategy.HYBRID_INTELLIGENT
    
    def optimize_query(self, query: str, entities: Optional[Dict[str, str]] = None) -> OptimizationResult:
        """
        Intelligently optimize query with optional GoT enhancement
        
        Args:
            query: The biomedical query to optimize and execute
            entities: Optional pre-extracted entities
            
        Returns:
            OptimizationResult with potential GoT enhancements
        """
        if not self.enable_got:
            return super().optimize_query(query, entities)
        
        # Handle async execution safely
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self._optimize_query_with_got_intelligence(query, entities))
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self._optimize_query_with_got_intelligence(query, entities))
    
    async def _optimize_query_with_got_intelligence(self, query: str, 
                                                  entities: Optional[Dict[str, str]] = None) -> OptimizationResult:
        """Intelligent query optimization with GoT integration"""
        start_time = time.time()
        
        # Analyze query complexity to determine strategy
        complexity = self._analyze_query_complexity(query, entities or {})
        logger.info(f"Query complexity analyzed as: {complexity}")
        
        # Try standard hybrid optimization first
        standard_result = await asyncio.to_thread(super().optimize_query, query, entities)
        
        # Decide whether to apply GoT based on result quality and complexity
        should_use_got = self._should_apply_got(standard_result, complexity)
        
        if should_use_got:
            logger.info(f"Applying GoT enhancement for {complexity} complexity query")
            self.complexity_got_usage[complexity] += 1
            
            try:
                # Apply GoT optimization
                got_result = await self._apply_got_optimization(
                    query, entities, complexity, standard_result
                )
                
                # Choose the better result
                final_result = self._select_best_result(standard_result, got_result)
                final_result.reasoning_chain.append(
                    f"Applied GoT enhancement for {complexity} complexity query"
                )
                
            except Exception as e:
                logger.warning(f"GoT enhancement failed: {e}")
                final_result = standard_result
                final_result.warnings.append(f"GoT enhancement failed: {str(e)}")
        else:
            logger.info(f"Standard optimization sufficient for {complexity} query")
            final_result = standard_result
            final_result.reasoning_chain.append(
                f"Standard optimization used for {complexity} complexity query"
            )
        
        # Add complexity and strategy information
        final_result.reasoning_chain.append(f"Query complexity: {complexity}")
        final_result.reasoning_chain.append(f"GoT used: {should_use_got}")
        
        execution_time = time.time() - start_time
        logger.info(f"Hybrid optimization completed in {execution_time:.2f}s, "
                   f"GoT applied: {should_use_got}")
        
        return final_result
    
    def _should_apply_got(self, standard_result: OptimizationResult, complexity: str) -> bool:
        """Determine whether to apply GoT based on result quality and complexity"""
        # Always use GoT for research complexity
        if complexity == 'research':
            return True
        
        # Use GoT for complex queries with suboptimal results
        if complexity == 'complex' and standard_result.metrics.quality_score < self.got_strategy_threshold:
            return True
        
        # Use GoT for moderate queries with poor results
        if complexity == 'moderate' and standard_result.metrics.quality_score < 0.4:
            return True
        
        # Use GoT if standard result has very few results
        if len(standard_result.results) < 3:
            return True
        
        return False
    
    async def _apply_got_optimization(self, query: str, entities: Optional[Dict[str, str]],
                                    complexity: str, baseline_result: OptimizationResult) -> OptimizationResult:
        """Apply GoT optimization with complexity-aware configuration"""
        # Configure GoT based on complexity
        config = self._get_complexity_aware_config(complexity)
        config['query'] = query
        
        # Execute GoT optimization
        got_result = await self.got_optimizer.optimize_with_got(
            query=query,
            strategy=self.get_strategy(),
            entities=entities,
            config=config
        )
        
        # Apply iterative refinement for complex queries
        if complexity in ['complex', 'research'] and got_result.success:
            logger.info(f"Applying iterative refinement for {complexity} query")
            got_result = await self.refinement_engine.refine_query_execution(
                got_result, {'query': query, 'complexity': complexity}
            )
        
        return got_result
    
    def _get_complexity_aware_config(self, complexity: str) -> Dict[str, Any]:
        """Get configuration parameters based on query complexity"""
        base_config = {
            'k': self.config.k if self.config else 5,
            'maxresults': self.config.max_results if self.config else 50
        }
        
        if complexity == 'simple':
            return base_config
        elif complexity == 'moderate':
            return {
                **base_config,
                'k': min(10, (self.config.k if self.config else 5) * 2),
                'maxresults': min(100, (self.config.max_results if self.config else 50) * 2)
            }
        elif complexity == 'complex':
            return {
                **base_config,
                'k': min(15, (self.config.k if self.config else 5) * 3),
                'maxresults': min(150, (self.config.max_results if self.config else 50) * 3)
            }
        else:  # research
            return {
                **base_config,
                'k': min(20, (self.config.k if self.config else 5) * 4),
                'maxresults': min(200, (self.config.max_results if self.config else 50) * 4)
            }
    
    def _select_best_result(self, standard_result: OptimizationResult, 
                          got_result: OptimizationResult) -> OptimizationResult:
        """Select the better result between standard and GoT optimization"""
        # Compare key metrics
        standard_score = self._calculate_overall_score(standard_result)
        got_score = self._calculate_overall_score(got_result)
        
        logger.info(f"Result comparison - Standard: {standard_score:.3f}, GoT: {got_score:.3f}")
        
        # Select better result with preference for GoT if close
        if got_score >= standard_score * 0.95:  # GoT preferred if within 5%
            return got_result
        else:
            return standard_result
    
    def _calculate_overall_score(self, result: OptimizationResult) -> float:
        """Calculate overall score for result comparison"""
        factors = [
            result.metrics.quality_score * 0.4,           # Quality (40%)
            min(1.0, len(result.results) / 20.0) * 0.3,   # Result count (30%)
            (1.0 if result.success else 0.0) * 0.2,       # Success (20%)
            min(1.0, len(result.entities) / 10.0) * 0.1   # Entity coverage (10%)
        ]
        return sum(factors)
    
    def get_complexity_usage_stats(self) -> Dict[str, int]:
        """Get statistics on GoT usage by query complexity"""
        return self.complexity_got_usage.copy()
    
    def get_hybrid_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for hybrid optimizer"""
        base_stats = self.get_strategy_performance_summary()
        
        return {
            "optimizer_type": "GoT-Enhanced Hybrid",
            "got_enabled": self.enable_got,
            "got_threshold": self.got_strategy_threshold,
            "strategy_performance": base_stats,
            "complexity_got_usage": self.complexity_got_usage,
            "total_got_applications": sum(self.complexity_got_usage.values())
        }


class GoTPerformanceComparator:
    """
    Utility class for comparing GoT performance against baseline methods
    """
    
    def __init__(self):
        self.comparison_results = []
    
    async def compare_optimizers(self, query: str, entities: Optional[Dict[str, str]] = None,
                               runs: int = 3) -> Dict[str, Any]:
        """
        Compare different optimizer configurations on the same query
        
        Args:
            query: Test query
            entities: Optional pre-extracted entities
            runs: Number of runs for statistical significance
            
        Returns:
            Comprehensive comparison results
        """
        logger.info(f"Running optimizer comparison for {runs} runs")
        
        # Initialize optimizers
        simple_baseline = SimpleWorkingOptimizer()
        simple_got = GoTEnhancedSimpleOptimizer(enable_got=True)
        hybrid_baseline = HybridIntelligentOptimizer()
        hybrid_got = GoTEnhancedHybridOptimizer(enable_got=True)
        
        optimizers = {
            'Simple Baseline': simple_baseline,
            'Simple + GoT': simple_got,
            'Hybrid Baseline': hybrid_baseline,
            'Hybrid + GoT': hybrid_got
        }
        
        results = {}
        
        # Run comparisons
        for name, optimizer in optimizers.items():
            logger.info(f"Testing {name}")
            
            run_results = []
            for run in range(runs):
                try:
                    if 'GoT' in name:
                        # For GoT optimizers, we need to handle async properly
                        result = optimizer.optimize_query(query, entities)
                    else:
                        result = optimizer.optimize_query(query, entities)
                    
                    run_results.append({
                        'execution_time': result.metrics.execution_time,
                        'quality_score': result.metrics.quality_score,
                        'result_count': len(result.results),
                        'entity_count': len(result.entities),
                        'success': result.success
                    })
                    
                except Exception as e:
                    logger.error(f"Run {run} failed for {name}: {e}")
                    run_results.append({
                        'execution_time': float('inf'),
                        'quality_score': 0.0,
                        'result_count': 0,
                        'entity_count': 0,
                        'success': False
                    })
            
            # Calculate statistics
            if run_results:
                results[name] = self._calculate_run_statistics(run_results)
                
                # Add GoT-specific metrics if available
                if hasattr(optimizer, 'get_got_performance_summary'):
                    results[name]['got_metrics'] = optimizer.get_got_performance_summary()
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(results)
        
        self.comparison_results.append({
            'query': query,
            'timestamp': time.time(),
            'results': results,
            'summary': comparison_summary
        })
        
        return {
            'query': query,
            'runs': runs,
            'results': results,
            'summary': comparison_summary
        }
    
    def _calculate_run_statistics(self, run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from multiple runs"""
        if not run_results:
            return {}
        
        # Filter successful runs for accurate statistics
        successful_runs = [r for r in run_results if r['success']]
        
        if not successful_runs:
            return {
                'success_rate': 0.0,
                'avg_execution_time': float('inf'),
                'avg_quality_score': 0.0,
                'avg_result_count': 0.0,
                'avg_entity_count': 0.0
            }
        
        return {
            'success_rate': len(successful_runs) / len(run_results),
            'avg_execution_time': sum(r['execution_time'] for r in successful_runs) / len(successful_runs),
            'avg_quality_score': sum(r['quality_score'] for r in successful_runs) / len(successful_runs),
            'avg_result_count': sum(r['result_count'] for r in successful_runs) / len(successful_runs),
            'avg_entity_count': sum(r['entity_count'] for r in successful_runs) / len(successful_runs),
            'min_execution_time': min(r['execution_time'] for r in successful_runs),
            'max_execution_time': max(r['execution_time'] for r in successful_runs)
        }
    
    def _generate_comparison_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary comparing all optimizers"""
        summary = {
            'best_quality': None,
            'fastest': None,
            'most_reliable': None,
            'quality_improvements': {},
            'speed_improvements': {}
        }
        
        if not results:
            return summary
        
        # Find best performer in each category
        best_quality_score = -1
        fastest_time = float('inf')
        best_success_rate = -1
        
        for name, stats in results.items():
            quality = stats.get('avg_quality_score', 0)
            exec_time = stats.get('avg_execution_time', float('inf'))
            success_rate = stats.get('success_rate', 0)
            
            if quality > best_quality_score:
                best_quality_score = quality
                summary['best_quality'] = name
            
            if exec_time < fastest_time and success_rate > 0:
                fastest_time = exec_time
                summary['fastest'] = name
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                summary['most_reliable'] = name
        
        # Calculate improvements
        baseline_names = [name for name in results.keys() if 'Baseline' in name]
        got_names = [name for name in results.keys() if 'GoT' in name]
        
        for got_name in got_names:
            # Find corresponding baseline
            baseline_name = got_name.replace(' + GoT', ' Baseline')
            if baseline_name in results:
                baseline_quality = results[baseline_name].get('avg_quality_score', 0)
                got_quality = results[got_name].get('avg_quality_score', 0)
                
                if baseline_quality > 0:
                    improvement = (got_quality - baseline_quality) / baseline_quality
                    summary['quality_improvements'][got_name] = improvement
                
                baseline_time = results[baseline_name].get('avg_execution_time', float('inf'))
                got_time = results[got_name].get('avg_execution_time', float('inf'))
                
                if baseline_time > 0 and got_time > 0:
                    speedup = baseline_time / got_time
                    summary['speed_improvements'][got_name] = speedup
        
        return summary
    
    def get_all_comparison_results(self) -> List[Dict[str, Any]]:
        """Get all comparison results"""
        return self.comparison_results.copy()
    
    def export_results_summary(self) -> Dict[str, Any]:
        """Export a comprehensive summary of all comparisons"""
        if not self.comparison_results:
            return {"message": "No comparison results available"}
        
        # Aggregate statistics across all comparisons
        total_comparisons = len(self.comparison_results)
        
        # Collect all quality improvements
        all_quality_improvements = []
        all_speed_improvements = []
        
        for comparison in self.comparison_results:
            summary = comparison.get('summary', {})
            quality_improvements = summary.get('quality_improvements', {})
            speed_improvements = summary.get('speed_improvements', {})
            
            all_quality_improvements.extend(quality_improvements.values())
            all_speed_improvements.extend(speed_improvements.values())
        
        return {
            "total_comparisons": total_comparisons,
            "avg_quality_improvement": sum(all_quality_improvements) / len(all_quality_improvements) if all_quality_improvements else 0,
            "avg_speed_improvement": sum(all_speed_improvements) / len(all_speed_improvements) if all_speed_improvements else 0,
            "quality_improvement_range": {
                "min": min(all_quality_improvements) if all_quality_improvements else 0,
                "max": max(all_quality_improvements) if all_quality_improvements else 0
            },
            "speed_improvement_range": {
                "min": min(all_speed_improvements) if all_speed_improvements else 0,
                "max": max(all_speed_improvements) if all_speed_improvements else 0
            },
            "detailed_results": self.comparison_results
        }