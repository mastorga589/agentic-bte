#!/usr/bin/env python3
"""
Limited demo test to verify the fixes work without running the full demo
"""

import asyncio
import logging
from agentic_bte.unified.examples.agent_demo import UnifiedAgentDemo
from agentic_bte.unified.agent import QueryMode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_limited_demo():
    """Test a few queries to verify the system works"""
    try:
        logger.info("🚀 Starting Limited Biomedical Agent Demo")
        
        demo = UnifiedAgentDemo()
        await demo.agent.initialize()
        await demo.setup_components()
        
        # Test queries
        test_queries = [
            ("What drugs treat diabetes?", QueryMode.STANDARD),
            ("How does metformin work?", QueryMode.BALANCED)
        ]
        
        for query_text, mode in test_queries:
            logger.info(f"\n📝 Processing: {query_text}")
            start_time = asyncio.get_event_loop().time()
            
            response = await demo.agent.process_query(
                text=query_text,
                query_mode=mode,
                max_results=5
            )
            
            end_time = asyncio.get_event_loop().time()
            
            logger.info(f"✅ Query completed in {end_time - start_time:.2f}s")
            logger.info(f"   Strategy: {str(response.strategy_used)}")
            logger.info(f"   Results: {response.total_results}")
            logger.info(f"   Stages: {len(response.stages_completed)}")
            
            if response.error:
                logger.error(f"   ❌ Error: {response.error}")
                return False
            else:
                logger.info(f"   ✅ Success with {response.total_results} results")
        
        logger.info("\n🎉 All queries completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False
    finally:
        if 'demo' in locals():
            await demo.agent.shutdown()

if __name__ == "__main__":
    success = asyncio.run(test_limited_demo())
    print(f"\n🏆 Demo {'PASSED' if success else 'FAILED'}")