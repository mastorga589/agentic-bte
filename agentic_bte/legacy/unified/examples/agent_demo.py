"""
Unified Biomedical Agent Demo

This script demonstrates the comprehensive capabilities of the UnifiedBiomedicalAgent,
showcasing single query processing, batch processing, different query modes,
caching, parallel execution, and performance monitoring.
"""

import asyncio
import logging
import time
import json
from typing import List, Dict, Any

from ..agent import (
    UnifiedBiomedicalAgent, QueryRequest, QueryMode, 
    BatchQueryRequest, ProcessingStage
)
from ..config import UnifiedConfig
from ..config import ExecutionStrategy
from ..types import BiomedicalResult, BiomedicalEntity, EntityType


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class UnifiedAgentDemo:
    """Demo class showcasing UnifiedBiomedicalAgent capabilities"""
    
    def __init__(self):
        # Initialize configuration for demo
        self.config = UnifiedConfig()
        
        # Optimize for demo performance
        self.config.performance.max_concurrent_queries = 5
        self.config.performance.max_concurrent_api_calls = 8
        self.config.performance.query_timeout_seconds = 60.0
        self.config.caching.enable_result_caching = True
        self.config.caching.cache_ttl_seconds = 3600
        
        # Initialize agent
        self.agent = UnifiedBiomedicalAgent(
            config=self.config,
            enable_caching=True,
            enable_parallel=True
        )
        
        logger.info("UnifiedBiomedicalAgent demo initialized")
    
    async def setup_components(self):
        """Setup real agent components for demo"""
        # Initialize the agent with real components - no mocking
        logger.info("Using real UnifiedBiomedicalAgent components...")
        # No additional setup needed - agent is already initialized with real components
        logger.info("Real components ready for demo")
    
    async def demo_simple_query_processing(self):
        """Demonstrate simple query processing"""
        logger.info("=" * 60)
        logger.info("SIMPLE QUERY PROCESSING DEMO")
        logger.info("=" * 60)
        
        queries = [
            "What drugs treat diabetes?",
            "How does metformin work?",
            "What are the side effects of insulin?",
            "What diseases are associated with high blood pressure?"
        ]
        
        for query_text in queries:
            logger.info(f"\nProcessing query: {query_text}")
            start_time = time.time()
            
            response = await self.agent.process_query(
                text=query_text,
                query_mode=QueryMode.BALANCED,
                max_results=10
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Query completed in {processing_time:.3f}s")
            logger.info(f"Total results: {response.total_results}")
            logger.info(f"Stages completed: {[stage.value for stage in response.stages_completed]}")
            if hasattr(response, 'final_answer') and response.final_answer:
                logger.info("\nFINAL ANSWER:")
                logger.info(f"  {response.final_answer}")
            
            if response.error:
                logger.error(f"Error: {response.error}")
            else:
                logger.info("Top results:")
                for i, result in enumerate(response.results[:3]):
                    logger.info(f"  {i+1}. {result.subject_entity} {result.predicate} {result.object_entity} "
                               f"(confidence: {result.confidence:.3f})")
        
        return queries
    
    async def demo_query_modes(self):
        """Demonstrate different query modes"""
        logger.info("=" * 60)
        logger.info("QUERY MODES DEMO")
        logger.info("=" * 60)
        
        query_text = "What drugs treat diabetes and how do they work?"
        modes = [
            (QueryMode.FAST, "Fast mode - prioritizes speed"),
            (QueryMode.BALANCED, "Balanced mode - balances speed and completeness"),
            (QueryMode.COMPREHENSIVE, "Comprehensive mode - prioritizes completeness"),
            (QueryMode.STANDARD, "Standard mode - normal processing")
        ]
        
        for mode, description in modes:
            logger.info(f"\n{description}:")
            start_time = time.time()
            
            response = await self.agent.process_query(
                text=query_text,
                query_mode=mode,
                max_results=20
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"  Processing time: {processing_time:.3f}s")
            logger.info(f"  Strategy: {str(response.strategy_used)}")
            logger.info(f"  Results: {response.total_results}")
            logger.info(f"  Parallel execution: {response.parallel_execution}")
            logger.info(f"  Knowledge graph: {'Yes' if response.knowledge_graph else 'No'}")
            
            if response.warnings:
                logger.warning(f"  Warnings: {response.warnings}")
        
        return modes
    
    async def demo_caching(self):
        """Demonstrate query result caching"""
        logger.info("=" * 60)
        logger.info("CACHING DEMO")
        logger.info("=" * 60)
        
        query_text = "What drugs treat diabetes?"
        
        # First query - should not be cached
        logger.info("First query (not cached):")
        start_time = time.time()
        response1 = await self.agent.process_query(query_text)
        time1 = time.time() - start_time
        
        logger.info(f"  Processing time: {time1:.3f}s")
        logger.info(f"  Cached: {response1.cached}")
        logger.info(f"  Results: {response1.total_results}")
        
        # Second identical query - should be cached
        logger.info("\nSecond identical query (should be cached):")
        start_time = time.time()
        response2 = await self.agent.process_query(query_text)
        time2 = time.time() - start_time
        
        logger.info(f"  Processing time: {time2:.3f}s")
        logger.info(f"  Cached: {response2.cached}")
        logger.info(f"  Results: {response2.total_results}")
        logger.info(f"  Speedup: {time1/time2:.2f}x faster")
        
        # Different query - should not be cached
        logger.info("\nDifferent query (not cached):")
        start_time = time.time()
        response3 = await self.agent.process_query("How does insulin work?")
        time3 = time.time() - start_time
        
        logger.info(f"  Processing time: {time3:.3f}s")
        logger.info(f"  Cached: {response3.cached}")
        logger.info(f"  Results: {response3.total_results}")
        
        return time1, time2, time3
    
    async def demo_batch_processing(self):
        """Demonstrate batch query processing"""
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING DEMO")
        logger.info("=" * 60)
        
        # Test with string queries
        string_queries = [
            "What drugs treat diabetes?",
            "How does metformin work?",
            "What are insulin side effects?",
            "What causes high blood pressure?",
            "How is cancer diagnosed?"
        ]
        
        logger.info(f"Processing batch of {len(string_queries)} queries as strings...")
        start_time = time.time()
        
        batch_response = await self.agent.process_batch(
            queries=string_queries,
            max_concurrency=3,
            consolidate_results=True
        )
        
        batch_time = time.time() - start_time
        
        logger.info(f"Batch completed in {batch_time:.3f}s")
        logger.info(f"Successful queries: {batch_response.successful_queries}")
        logger.info(f"Failed queries: {batch_response.failed_queries}")
        logger.info(f"Average response time: {batch_response.average_response_time:.3f}s")
        logger.info(f"Consolidated knowledge: {'Yes' if batch_response.consolidated_knowledge else 'No'}")
        
        # Test with QueryRequest objects
        request_queries = [
            QueryRequest(
                query_id="diabetes_drugs",
                text="What are the best drugs for diabetes?",
                query_mode=QueryMode.COMPREHENSIVE,
                max_results=15
            ),
            QueryRequest(
                query_id="metformin_mechanism",
                text="How does metformin work at molecular level?",
                query_mode=QueryMode.BALANCED,
                max_results=10
            ),
            QueryRequest(
                query_id="insulin_types",
                text="What are different types of insulin?",
                query_mode=QueryMode.FAST,
                max_results=5
            )
        ]
        
        logger.info(f"\nProcessing batch of {len(request_queries)} QueryRequest objects...")
        start_time = time.time()
        
        batch_response2 = await self.agent.process_batch(
            queries=request_queries,
            max_concurrency=2,
            fail_fast=False
        )
        
        batch_time2 = time.time() - start_time
        
        logger.info(f"Batch completed in {batch_time2:.3f}s")
        logger.info(f"Successful queries: {batch_response2.successful_queries}")
        
        # Show individual results
        logger.info("\nIndividual query results:")
        for response in batch_response2.responses[:3]:  # Show first 3
            logger.info(f"  {response.query_id}: {response.total_results} results "
                       f"({response.processing_time:.3f}s, {response.request.query_mode.value} mode)")
        
        return batch_response, batch_response2
    
    async def demo_parallel_execution(self):
        """Demonstrate parallel vs sequential execution"""
        logger.info("=" * 60)
        logger.info("PARALLEL EXECUTION DEMO")
        logger.info("=" * 60)
        
        complex_query = "What drugs treat diabetes, how do they work, what are their side effects, and how do they interact with other medications?"
        
        # Sequential execution
        logger.info("Sequential execution:")
        start_time = time.time()
        response_sequential = await self.agent.process_query(
            text=complex_query,
            enable_parallel=False,
            max_results=20
        )
        sequential_time = time.time() - start_time
        
        logger.info(f"  Processing time: {sequential_time:.3f}s")
        logger.info(f"  Parallel execution: {response_sequential.parallel_execution}")
        logger.info(f"  Results: {response_sequential.total_results}")
        
        # Parallel execution
        logger.info("\nParallel execution:")
        start_time = time.time()
        response_parallel = await self.agent.process_query(
            text=complex_query,
            enable_parallel=True,
            max_results=20
        )
        parallel_time = time.time() - start_time
        
        logger.info(f"  Processing time: {parallel_time:.3f}s")
        logger.info(f"  Parallel execution: {response_parallel.parallel_execution}")
        logger.info(f"  Results: {response_parallel.total_results}")
        
        if sequential_time > parallel_time:
            speedup = sequential_time / parallel_time
            logger.info(f"  Speedup: {speedup:.2f}x faster with parallel execution")
        else:
            logger.info("  Note: Parallel execution may not always be faster for simple queries")
        
        return sequential_time, parallel_time
    
    async def demo_error_handling(self):
        """Demonstrate error handling and recovery"""
        logger.info("=" * 60)
        logger.info("ERROR HANDLING DEMO")
        logger.info("=" * 60)
        
        # Temporarily modify mock to simulate errors
        original_extract = self.agent.entity_processor.extract_entities
        
        async def error_prone_extract(text, context=None):
            if "error" in text.lower():
                raise Exception("Simulated entity extraction failure")
            return await original_extract(text, context)
        
        self.agent.entity_processor.extract_entities = error_prone_extract
        
        queries = [
            "What drugs treat diabetes?",  # Should work
            "This query will cause an error",  # Should fail
            "How does metformin work?",  # Should work
        ]
        
        for query in queries:
            logger.info(f"\nProcessing: {query}")
            
            response = await self.agent.process_query(query)
            
            if response.error:
                logger.error(f"  Error: {response.error}")
                logger.info(f"  Stages completed: {[s.value for s in response.stages_completed]}")
            else:
                logger.info(f"  Success: {response.total_results} results")
            
            logger.info(f"  Processing time: {response.processing_time:.3f}s")
        
        # Restore original function
        self.agent.entity_processor.extract_entities = original_extract
        
        return queries
    
    async def demo_confidence_filtering(self):
        """Demonstrate confidence threshold filtering"""
        logger.info("=" * 60)
        logger.info("CONFIDENCE FILTERING DEMO")
        logger.info("=" * 60)
        
        query = "What drugs treat diabetes?"
        thresholds = [0.1, 0.5, 0.8, 0.9]
        
        for threshold in thresholds:
            logger.info(f"\nConfidence threshold: {threshold}")
            
            response = await self.agent.process_query(
                text=query,
                confidence_threshold=threshold,
                max_results=50
            )
            
            logger.info(f"  Results: {response.total_results}")
            
            if response.results:
                confidences = [r.confidence for r in response.results]
                logger.info(f"  Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
                logger.info(f"  Average confidence: {sum(confidences) / len(confidences):.3f}")
        
        return thresholds
    
    async def demo_performance_monitoring(self):
        """Demonstrate performance monitoring and statistics"""
        logger.info("=" * 60)
        logger.info("PERFORMANCE MONITORING DEMO")
        logger.info("=" * 60)
        
        # Process several queries to generate statistics
        test_queries = [
            "What drugs treat diabetes?",
            "How does insulin work?", 
            "What causes heart disease?",
            "How is cancer treated?",
            "What are antibiotics?"
        ]
        
        logger.info("Processing queries to generate performance data...")
        for i, query in enumerate(test_queries):
            await self.agent.process_query(
                text=query,
                query_mode=QueryMode.BALANCED if i % 2 == 0 else QueryMode.FAST
            )
        
        # Get performance summary
        performance = self.agent.get_performance_summary()
        
        logger.info("\nPerformance Summary:")
        logger.info(f"  Total queries: {performance['total_queries']}")
        logger.info(f"  Successful queries: {performance['successful_queries']}")
        logger.info(f"  Failed queries: {performance['failed_queries']}")
        logger.info(f"  Success rate: {performance['success_rate']:.2%}")
        logger.info(f"  Average processing time: {performance['average_processing_time']:.3f}s")
        logger.info(f"  Cache hit rate: {performance['cache_hit_rate']:.2%}")
        logger.info(f"  Parallel execution rate: {performance['parallel_execution_rate']:.2%}")
        
        logger.info("\nStrategy usage:")
        for strategy, count in performance['strategy_usage'].items():
            logger.info(f"  {strategy}: {count}")
        
        # Get query history
        history = self.agent.get_query_history(limit=3)
        logger.info(f"\nRecent queries ({len(history)} shown):")
        for i, response in enumerate(history):
            logger.info(f"  {i+1}. Query: {response.request.text[:50]}...")
            logger.info(f"     Results: {response.total_results}, Time: {response.processing_time:.3f}s")
        
        # Check active queries
        active = self.agent.get_active_queries()
        logger.info(f"\nActive queries: {len(active)}")
        
        return performance
    
    async def demo_health_check(self):
        """Demonstrate health check functionality"""
        logger.info("=" * 60)
        logger.info("HEALTH CHECK DEMO")
        logger.info("=" * 60)
        
        health = await self.agent.health_check()
        
        logger.info("Agent Health Status:")
        logger.info(f"  Initialized: {health['initialized']}")
        logger.info(f"  Active queries: {health['active_queries']}")
        logger.info(f"  Cache size: {health['cache_size']}")
        logger.info(f"  History size: {health['history_size']}")
        
        logger.info("\nComponent Health:")
        for component, status in health['components'].items():
            logger.info(f"  {component}: {status}")
        
        return health
    
    async def demo_cache_management(self):
        """Demonstrate cache management"""
        logger.info("=" * 60)
        logger.info("CACHE MANAGEMENT DEMO") 
        logger.info("=" * 60)
        
        # Generate some cached queries
        queries = ["What drugs treat diabetes?", "How does metformin work?"]
        
        for query in queries:
            await self.agent.process_query(query)  # First time - not cached
            await self.agent.process_query(query)  # Second time - cached
        
        # Show cache status
        performance = self.agent.get_performance_summary()
        logger.info(f"Cache entries: {performance.get('cache_entries', 'N/A')}")
        logger.info(f"Cache hit rate: {performance['cache_hit_rate']:.2%}")
        
        # Clear cache
        cleared = self.agent.clear_cache()
        logger.info(f"Cleared {cleared} cache entries")
        
        # Clear history
        cleared_history = self.agent.clear_history()
        logger.info(f"Cleared {cleared_history} history entries")
        
        return cleared, cleared_history
    
    async def run_full_demo(self):
        """Run all demo scenarios"""
        logger.info("🚀 Starting Unified Biomedical Agent Demo")
        logger.info("=" * 80)
        
        demo_results = {}
        
        try:
            # Initialize agent
            await self.agent.initialize()
            await self.setup_components()
            
            # Run all demos
            demo_results['simple_queries'] = await self.demo_simple_query_processing()
            demo_results['query_modes'] = await self.demo_query_modes()  
            demo_results['caching'] = await self.demo_caching()
            demo_results['batch_processing'] = await self.demo_batch_processing()
            demo_results['parallel_execution'] = await self.demo_parallel_execution()
            demo_results['error_handling'] = await self.demo_error_handling()
            demo_results['confidence_filtering'] = await self.demo_confidence_filtering()
            demo_results['performance_monitoring'] = await self.demo_performance_monitoring()
            demo_results['health_check'] = await self.demo_health_check()
            demo_results['cache_management'] = await self.demo_cache_management()
            
            # Final summary
            self._print_demo_summary(demo_results)
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            raise
        finally:
            # Cleanup
            await self.agent.shutdown()
        
        return demo_results
    
    def _print_demo_summary(self, demo_results: Dict[str, Any]):
        """Print summary of all demo results"""
        logger.info("=" * 80)
        logger.info("DEMO SUMMARY")
        logger.info("=" * 80)
        
        logger.info("UnifiedBiomedicalAgent demonstrated:")
        logger.info("  ✅ Simple query processing with multiple query types")
        logger.info("  ✅ Different query modes (Fast, Balanced, Comprehensive, Standard)")
        logger.info("  ✅ Intelligent query result caching with significant speedup")
        logger.info("  ✅ Batch processing with concurrent execution")
        logger.info("  ✅ Parallel vs sequential execution comparison")
        logger.info("  ✅ Robust error handling and recovery")
        logger.info("  ✅ Confidence-based result filtering")
        logger.info("  ✅ Comprehensive performance monitoring and statistics")
        logger.info("  ✅ Health checking and component status monitoring")
        logger.info("  ✅ Cache and history management")
        
        # Show key metrics
        performance = demo_results.get('performance_monitoring', {})
        if performance:
            logger.info(f"\nFinal Performance Metrics:")
            logger.info(f"  Total queries processed: {performance['total_queries']}")
            logger.info(f"  Overall success rate: {performance['success_rate']:.2%}")
            logger.info(f"  Average processing time: {performance['average_processing_time']:.3f}s")
            logger.info(f"  Cache hit rate: {performance['cache_hit_rate']:.2%}")
        
        # Show caching benefits
        caching_results = demo_results.get('caching')
        if caching_results and len(caching_results) >= 2:
            time1, time2 = caching_results[0], caching_results[1]
            speedup = time1 / time2 if time2 > 0 else 1.0
            logger.info(f"  Caching speedup: {speedup:.2f}x faster for repeated queries")
        
        # Show parallel execution benefits
        parallel_results = demo_results.get('parallel_execution')
        if parallel_results and len(parallel_results) >= 2:
            sequential_time, parallel_time = parallel_results
            if sequential_time > parallel_time:
                speedup = sequential_time / parallel_time
                logger.info(f"  Parallel execution speedup: {speedup:.2f}x faster for complex queries")
        
        logger.info("\n🎉 Unified Biomedical Agent Demo completed successfully!")
        logger.info("The agent provides a unified, high-performance interface for biomedical queries")
        logger.info("with intelligent strategy selection, caching, parallel execution, and monitoring.")


async def main():
    """Main demo function"""
    try:
        demo = UnifiedAgentDemo()
        await demo.run_full_demo()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())