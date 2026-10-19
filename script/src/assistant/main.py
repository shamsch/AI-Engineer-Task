import sys

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart

from assistant.brochures import list_products, search_brochures
from assistant.config import MODEL
from assistant.services import estimate_service_cost

INSTRUCTIONS = """You are a sales assistant for Company M Oy salespeople who are on a call with mining customers.

You can:
1. Answer product questions and recommend a product, using ONLY the brochures (tools: list_products, search_brochures).
2. Estimate the cost of a field service visit, using ONLY historical service reports (tool: estimate_service_cost).

Rules:
- Decide per message which tool(s) to call. Do not answer product or cost questions from general knowledge.
- Every product fact or number must come from a tool result. In `sources`, list only the brochure pages and reports that support facts in your answer, formatted exactly as "<brochure> p.<page>" (e.g. "larox-pf-ds p.2") or "FSR-XXXX". A clarifying question has no sources.
- If the data does not contain the answer, say so plainly. Never guess specifications, prices, costs or capabilities, and do not speculate about factors the data does not cover.
- If a required input is missing and it would change the answer, ask ONE short clarifying question instead of guessing. Otherwise do not end with a question.
- When recommending, name exactly one product and tie the reason to what the customer said. Mention alternatives only if asked.
- If sources conflict or look inconsistent, point it out.
- Never call estimate_service_cost unless the user stated the equipment; ask first. If the work needed is missing, ask for it too. Give the budget range per currency, on-site and travel hours (use travel_hours_to_country_any_equipment when the country filter was relaxed), the reports used, and any relaxed filters or gaps.
- Never offer actions you cannot perform (quotes, contacting teams, currency conversion, follow-ups). No next-step offers.
- Keep it short: at most 5 bullets or about 80 words, no preamble. The salesperson is on a live call."""


class Answer(BaseModel):
    answer: str
    sources: list[str]


agent = Agent(MODEL, output_type=Answer, instructions=INSTRUCTIONS, retries=3, tools=[list_products, search_brochures, estimate_service_cost])


@agent.output_validator
def check_citations(ctx: RunContext[None], output: Answer) -> Answer:
    retrieved: set[str] = set()
    for msg in ctx.messages:
        for part in msg.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name in ("search_brochures", "list_products"):
                retrieved |= {f"{r['brochure']} p.{r.get('page', 1)}" for r in part.content}
            elif isinstance(part, ToolReturnPart) and part.tool_name == "estimate_service_cost":
                retrieved |= set(part.content.get("matched_reports", []))
    unverified = [s for s in output.sources if s not in retrieved]
    if unverified:
        raise ModelRetry(f"These sources were never retrieved by a tool: {unverified}. Cite only retrieved sources.")
    return output


def print_tool_calls(messages: list[ModelMessage]) -> None:
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    print(f"  [tool] {part.tool_name}({part.args_as_json_str()})")


def main() -> None:
    debug = "--debug" in sys.argv
    history: list[ModelMessage] = []
    print(f"Company M Oy sales assistant ({MODEL}). Type 'exit' to quit.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        result = agent.run_sync(user, message_history=history)
        if debug:
            print_tool_calls(result.new_messages())
        history = result.all_messages()
        print(f"\nassistant> {result.output.answer}")
        if result.output.sources:
            print(f"Sources: {', '.join(result.output.sources)}")
