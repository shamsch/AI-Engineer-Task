import os

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

from assistant.main import agent
from assistant.services import CACHE

pytestmark = pytest.mark.skipif(os.getenv("RUN_EVALS") != "1", reason="real API calls; set RUN_EVALS=1")


def tool_calls(messages: list[ModelMessage]) -> list[ToolCallPart]:
    return [p for m in messages if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart)]

# tests that it makes use of the right tool call
def test_product_recommendation() -> None:
    result = agent.run_sync("We have Larox pressure filters but performance drops whenever our feed grade changes")
    assert "search_brochures" in {c.tool_name for c in tool_calls(result.new_messages())}
    assert result.output.sources

# tests that it identify right eqipment and pulls the correct file in citaiton
@pytest.mark.skipif(not CACHE.exists(), reason="run `uv run ingest` first")
def test_cost_estimate() -> None:
    result = agent.run_sync(
        "A customer in Chile needs 12 filter cloths replaced on a Larox PF-DS tower press. What should I budget for the service visit?"
    )
    calls = [c for c in tool_calls(result.new_messages()) if c.tool_name == "estimate_service_cost"]
    assert any(c.args_as_dict()["equipment"] == "Larox PF-DS" for c in calls)
    assert any(s.startswith("FSR-") for s in result.output.sources)


# when equipement is not given, make sure the tool call was made with "other" equipment, not invent an equipment
def test_vague_cost_question_asks_for_info() -> None:
    result = agent.run_sync("What would a service visit cost?")
    calls = [c for c in tool_calls(result.new_messages()) if c.tool_name == "estimate_service_cost"]
    assert all(c.args_as_dict()["equipment"] == "other" for c in calls)
    assert "?" in result.output.answer


def test_unknown_product_not_in_data() -> None:
    result = agent.run_sync("What is the price of the Orion 9000 pump?")
    assert any(w in result.output.answer.lower() for w in ("not", "n't", "n’t", "cannot", "no "))
