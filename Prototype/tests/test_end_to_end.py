import os
import asyncio
import pytest

pytestmark = pytest.mark.skipif(
    'OPENAI_API_KEY' not in os.environ,
    reason="OPENAI_API_KEY is not set; end-to-end tests require real LLM calls"
)


def _run(query: str) -> str:
    from Prototype.got_planner import GoTPlanner
    planner = GoTPlanner()
    return asyncio.run(planner.execute(query))


@pytest.mark.parametrize(
    "query",
    [
        "Which small molecules can treat Crohn's disease by modulating the immune response?",
        "Which genes are associated with Parkinson's disease and are targets of small molecules?",
        "Which small molecules modulate the PI3K/AKT signaling pathway involved in breast cancer?",
    ],
)
def test_got_planner_complex_queries(query):
    answer = _run(query)
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
