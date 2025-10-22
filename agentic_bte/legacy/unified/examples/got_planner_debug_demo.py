import asyncio
from agentic_bte.unified.got_planner import GoTPlanner, GPT41GoTLLM

async def run_llm_planning(query):
    planner = GoTPlanner(GPT41GoTLLM())
    print("Running GoTPlanner decomposition...")
    try:
        # Use the public decompose_initial_query method which handles parsing internally
        plan = await planner.decompose_initial_query(query)
        print("\n==== DECOMPOSED QUERY PLAN ====\n", plan)
        
        # Optionally, also show raw LLM output for debugging
        print("\n==== Getting raw LLM output for comparison ====")
        raw_output = await planner.llm.decompose_to_graph(query)
        print("\n==== RAW LLM OUTPUT ====\n", raw_output, flush=True)
        
    except Exception as e:
        print(f"\n[DEBUG] Exception during GoT planning: {e}")

if __name__ == "__main__":
    query = "Which drugs can treat Spinal muscular atrophy by targeting Alternative mRNA splicing, via spliceosome?"
    asyncio.run(run_llm_planning(query))
