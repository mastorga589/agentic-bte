"""
Smoke test for full agentic-bte pipeline: GoT/NER/Knowledge/Result
"""
import asyncio
from agentic_bte.unified.execution_engine import UnifiedExecutionEngine
from agentic_bte.unified.config import UnifiedConfig

async def main():
    config = UnifiedConfig()  # Use default config for now
    engine = UnifiedExecutionEngine(config)
    query = "What drugs treat diabetes?"
    print(f"\n[RUNNING QUERY]: {query}\n")
    result = await engine.execute_query(query)
    print("\n=== PIPELINE RESULT ===")
    print(f"Success: {result.success}")
    print(f"Final answer: {result.final_answer}\n")
    print(f"Entities: {getattr(result, 'entities', {})}")
    print(f"Predicted relationships: {getattr(result, 'relationships', [])}")
    print(f"Pipeline reasoning chain:")
    for step in getattr(result, 'reasoning_chain', []):
        print("    -", step)
    if result.errors:
        print("ERRORS:", result.errors)
    if result.warnings:
        print("WARNINGS:", result.warnings)

if __name__ == "__main__":
    asyncio.run(main())
