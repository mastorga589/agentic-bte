import asyncio
from agentic_bte.unified.got_planner import GPT41GoTLLM

def raw_llm_plan(query: str):
    async def run():
        planner_llm = GPT41GoTLLM()
        from agentic_bte.unified.got_planner import LANGGRAPH_PLANNING_PROMPT
        prompt = LANGGRAPH_PLANNING_PROMPT.format(query=query)
        result = await planner_llm.llm.ainvoke([{"role": "user", "content": prompt}])
        print("\nRAW LLM OUT:\n" + result.content)
    asyncio.run(run())

if __name__ == "__main__":
    query = "Which drugs can treat Spinal muscular atrophy by targeting Alternative mRNA splicing, via spliceosome?"
    raw_llm_plan(query)
