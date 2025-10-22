import os
import pytest

pytestmark = pytest.mark.skipif(
    'OPENAI_API_KEY' not in os.environ,
    reason="OPENAI_API_KEY is not set; end-to-end tests require real LLM calls"
)


def test_langgraph_agent_end_to_end():
    from Prototype.Agent import BTEx
    q = "Which small molecules modulate the PI3K/AKT signaling pathway involved in breast cancer?"
    ans = BTEx(q, maxresults=50, k=5)
    assert isinstance(ans, str)
    assert len(ans.strip()) > 0
