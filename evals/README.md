# Eval

Measures how well the agent turns a chat message into the right PayPal calls, and
compares ways of choosing the tool.

## What is checked

Two levels:

- **Selection** (every case, no API calls). Intent; the operation for each goal in the
  message; which IDs are new; which IDs the user typed; whether a calculation is needed;
  whether the agent asked when it should. For single-goal cases with a `plan`, the
  planned call sequence too. Messages are split into ordered actions first, as the agent
  does (the all-tools control never splits). A single-goal chain case that the agent
  splits ("create a product, then a plan for it, then subscribe sam") counts as right
  when the calls its actions plan, in order, are exactly the expected `plan` and the last
  action is the goal; typed IDs and the calculation flag are still checked, on the last
  action (the goal step the case's expectation describes); see `score.score_split_chain`.
- **End to end on the mock** (cases with an `e2e` block). The status after the first
  message; that nothing is sent before approval; key fields in the planned request
  bodies and queries; the calls made after approving; the answer (computed totals,
  RAG sources, system search kind).

Variants compared at the selection level (everything after selection is the same agent):

| Variant | Who chooses the tool |
|---|---|
| `jev` | Jev, with the LLM as fallback. The agent as shipped. |
| `jev-only` | Jev, asking the user instead of falling back. |
| `llm` | The LLM: resource first, then action. |
| `all-tools` | Control: every tool in one LLM prompt, the approach the task says breaks down. |

## Method

- Each category is split in half: thresholds and prompts are tuned on `tune`, and the
  numbers to report come from `test`.
- `cases_user.jsonl` holds additional cases; they are split the same way.
- `cases_external.jsonl` holds the 12 router cases published by another submission
  (Sudharsan2618/datazoic-task), mapped to our operation IDs. They are reported in their
  own table and are not used for tuning.
- Model and Jev answers are cached (`cache.sqlite`), so a re-run of unchanged code
  gives identical numbers. `--no-cache` asks again, to see how much answers vary.

## Running

    python evals/run_eval.py                         all variants, both levels
    python evals/run_eval.py --variants jev-only     one variant
    python evals/run_eval.py --level selection       skip end to end
    python evals/run_eval.py --only m01,multi_action a few cases or categories
    python evals/run_eval.py --langsmith             also log LangSmith experiments

Results are written to `results/<time>/summary.md` (tables and every failure) and
`results/<time>/results.json`. The LLM variants are slow on Groq's free tier
(8,000 tokens a minute); the cache makes later runs fast.

## Scaling eval

`run_scale.py` runs the selection eval on four catalogues, adding other vendors' public
OpenAPI specs as distractors and as new targets:

| Tools | Vendors | Source |
|---|---|---|
| 110 | PayPal | Postman collection, reviewed labels |
| 672 | + Stripe (562) | `stripe/openapi` spec3.json, tag v2599 |
| 977 | + Square (305) | `square/connect-api-specification` api.json |
| 2,220 | + GitHub (1,243) | `github/rest-api-description` api.github.com.json |

    python scripts/fetch_specs.py                          download the pinned specs (git-ignored)
    python build_openapi.py                                build data/specs/<vendor>_model.json
    python evals/run_scale.py --size 2220 --variants jev --out evals/results/scale-x
    python evals/run_scale.py --report evals/results/scale-x

- The added vendors' labels (resource, action, ID links, risk tier) are inferred by fixed
  rules in `build_openapi.py`, not reviewed by a person; their ID links are marked inferred.
- PayPal is set as the user's main account, so the 96 PayPal cases, which name no vendor,
  still mean PayPal. The cases in `cases_scale.jsonl` name their vendor or ask for something
  only one vendor does.
- Variants: `jev`, `llm`, `all-tools`, and `retrieval@10` (the langgraph-bigtool approach:
  embed every tool card with text-embedding-3-small, keep the 10 nearest to the message, let
  the LLM pick among them). Recall@10 is reported for the retrieval baseline.
- The scaling eval never uses the cache.

## Writing a case

One JSON object per line in `cases_user.jsonl`; the file's header explains the
fields. Write the message the way a real user would, then the right outcome. Use
`python demo.py "your message"` to see operation IDs.
