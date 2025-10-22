#!/usr/bin/env python3
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

from agentic_bte.unified.agent import (
    UnifiedBiomedicalAgent, QueryRequest, QueryMode, 
    BatchQueryRequest, ProcessingStage
)
from agentic_bte.unified.config import UnifiedConfig, ExecutionStrategy
from agentic_bte.unified.types import BiomedicalResult, BiomedicalEntity, EntityType


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


async def main():
    """Run the unified agent demo"""
    demo = UnifiedAgentDemo()
    
    try:
        await demo.demo_simple_query_processing()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())