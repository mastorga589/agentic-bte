import asyncio
from agentic_bte.unified.agent import UnifiedBiomedicalAgent

def print_final_answer(response):
    print("\n--- FINAL ANSWER ---")
    print(response.final_answer or "[No answer available]")
    print("--- RAW OUTPUT ---")
    print("Results:", response.total_results)
    print("Stages:", response.stages_completed)
    print()

async def main():
    agent = UnifiedBiomedicalAgent()
    await agent.initialize()
    print("Running GoTPlanner query...")
    query = "What drugs treat diabetes?"
    response = await agent.process_query(
        text=query,
        query_mode=None,
        max_results=10
    )
    print_final_answer(response)
    await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
