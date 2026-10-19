# Sales Assistant Agent

A small CLI agent for a salesperson on a customer call. It answers product questions from the brochures and estimates service visit costs from past field reports. If something is missing it asks, and if the data doesn't have the answer it says so.

Covers parts 1 (products) and 2 (service costs). The optional part 3 (flotation process data) was skipped.

## How it works

```
message -> PydanticAI agent (LLM picks a tool) -> answer {text, sources} -> citation check
              |- list_products / search_brochures   brochures, BM25 per page
              |- estimate_service_cost              past reports, plain-code statistics
```

- **Routing**: no hand-written router. The model reads the tool descriptions and picks one (or several) per message. `--debug` prints every tool call.
- **Brochures** (`brochures.py`): each brochure is split by page and searched with BM25. The model is told to rephrase its query 2-3 times, because brochure wording rarely matches the customer's.
- **Service costs** (`services.py`): the 50 reports are messy (tables, free text, missing costs, 5 currencies), so `uv run ingest` turns each one into a typed record with an LLM, once. Missing values stay null, nothing is guessed. The cost tool then filters by equipment, task and country, drops country and then task if nothing matches (and says so), and returns ranges per currency. No currency conversion, because rates aren't in the data.
- **Grounding** (`main.py`): answers come back as `{answer, sources}`. A validator rejects any cited page or report that no tool actually returned, and the model has to retry.
- **Tests** (`tests/`): the brief's prompts plus edge cases, checked on behaviour (which tool, what was cited), not on wording.

> [!NOTE]
> **What is BM25?** Classic keyword ranking, what search engines used before embeddings. A page scores higher when it contains the query words, more so for rare words like "PF-DS", and long pages are penalised so they don't win just by being long. No embedding API, no vector database, and you can see why a page ranked.

## Run

Needs [uv](https://docs.astral.sh/uv/). Put the provided `data/` folder next to `script/`.

```bash
cd script
cp .env.example .env            # set OPENROUTER_API_KEY
uv sync
uv run ingest                   # one-off, ~1 min: reports -> .cache/services.json
uv run assistant --debug        # chat
RUN_EVALS=1 uv run pytest -q    # live evals (real model calls)
```

Any PydanticAI model string works in `MODEL`, e.g. `openai:gpt-5-mini` with `OPENAI_API_KEY`.



## Known limits

- The citation check proves a source was retrieved, not that it supports the claim.
- Asking for missing info relies on the prompt. The cost tool refuses `equipment="other"` as a backstop.
## What I'd do next

- **Router step**: a cheap first call returning `intent` and `missing_info`, so clarifying questions are enforced in code. Skipped because three tools is well within plain tool calling.
- **LLM-as-judge** to check that each claim is actually supported by its source.
- **Hybrid search** (BM25 + embeddings, reranking) once there are more brochures.
- **Guardrail frameworks** (Guardrails AI, NeMo Guardrails) for policy checks.
