import os
import asyncio

def test_demo_print_complex_query():
    from Prototype.got_planner import GoTPlanner
    query = "Which small molecules modulate the PI3K/AKT signaling pathway involved in breast cancer?"
    print("\n[DEMO] Query:", query)
    ans = asyncio.run(GoTPlanner().execute(query))
    print("\n[DEMO] Final answer:\n", ans)
    assert isinstance(ans, str) and ans.strip(), "Empty final answer"
