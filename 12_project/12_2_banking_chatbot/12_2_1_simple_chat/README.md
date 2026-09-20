# Step 1 — Naive Agent: One Box, Too Much Responsibility

## Files changed

- `12_2_1_simple_chat.py` (new) — the entire pipeline for this step.

## Architecture change

A single LangGraph node wrapping a bare `ChatOpenAI` call:

```
START -> generate -> END
```

No tools, no authentication, no memory, no RAG. The agent can only answer from its
LLM training knowledge — it has no connection to any bank system. This is
intentionally the "wrong" architecture: every later step in this build exists to
close a gap this step exposes.

## Demo prompt

```
What is my account balance?
```

## Expected output

The agent admits it has no access to real account data — it must not invent a
number. Something like: *"I don't have access to your bank account information."*

## Failure case

If the model instead states a specific balance figure (e.g. "$4,231.50"), that is a
hallucination — this step has no tool and no data source, so any concrete figure in
the response is fabricated. That failure is exactly why step 2 adds a real tool.

## Regression test

None — this is the first step.

## Run

```
python 12_2_1_simple_chat.py
```
