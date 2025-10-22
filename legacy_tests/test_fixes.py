#!/usr/bin/env python3
"""
Simple test script to verify the fixes work
"""

import asyncio
import logging
from agentic_bte.unified.examples.agent_demo import UnifiedAgentDemo
from agentic_bte.unified.agent import QueryMode

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_simple_query():
    """Test a simple query to verify fixes work"""
    try:
        logger.info("Initializing demo...")
        demo = UnifiedAgentDemo()
        
        logger.info("Initializing agent...")
        await demo.agent.initialize()
        await demo.setup_components()
        
        logger.info("Processing test query...")
        response = await demo.agent.process_query(
            text="What drugs treat diabetes?",
            query_mode=QueryMode.STANDARD, 
            max_results=5
        )
        
        logger.info(f"Query completed successfully!")
        logger.info(f"Strategy used: {response.strategy_used}")
        logger.info(f"Total results: {response.total_results}")
        logger.info(f"Processing time: {response.processing_time:.2f}s")
        logger.info(f"Stages: {[s.value if hasattr(s, 'value') else str(s) for s in response.stages_completed]}")
        
        if response.error:
            logger.error(f"Query had error: {response.error}")
            return False
        else:
            logger.info("✅ Query processed successfully!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    finally:
        if 'demo' in locals():
            await demo.agent.shutdown()

if __name__ == "__main__":
    success = asyncio.run(test_simple_query())
    print(f"Test {'PASSED' if success else 'FAILED'}")