# Multi-tool API agent

A chat agent that picks the right API calls, with the right parameters, from a large
catalogue, and keeps doing so as the catalogue grows. A model never writes a tool
name. It makes small, scored choices among real options, and code orders, checks,
approves and executes the calls.

Built on the PayPal Postman collection (110 tools). It is also measured at 2,220 tools,
with the Stripe, Square and GitHub API specs added.

| At 110 tools (held-out cases, no cache) | Fully right | Typical time |
|---|---|---|
| Agent as shipped (Jev, LLM agent as fallback) | 98% | 1.3 s |
| LLM agent alone (resource, then operation) | 100% | 5.0 s |
| LLM agent with all 110 tools in one prompt (control) | 66% | 3.0 s |
| Live PayPal sandbox, end to end | 11 of 11 cases passed | - |

## Scaling to 2,220 tools

As the catalogue grows twentyfold, the agent as shipped stays at 95–100% on the PayPal
cases. One prompt with every tool stays near 60%. Top-10 embedding retrieval (the
langgraph-bigtool approach) falls to 51%, because its recall drops to 79% once other
vendors are added.

| PayPal cases fully right (96, no cache) | 110 tools | 672 (+Stripe) | 977 (+Square) | 2,220 (+GitHub) |
|---|---|---|---|---|
| Agent as shipped (Jev, LLM agent as fallback) | 100% | 97% | 97% | 95% |
| LLM agent alone | 97% | 92% | 93% | 94% |
| LLM agent with every tool in one prompt | 62% | 60% | 64% | 59% |
| Retrieval top 10, then the LLM agent picks | 66% | 56% | 57% | 51% |
| Retrieval recall@10 | 98% | 80% | 79% | 79% |

| At 2,220 tools | New-vendor cases fully right (36) | Tokens per PayPal request, 110 → 2,220 tools |
|---|---|---|
| Agent as shipped | 97% | 3,804 → 6,246 |
| LLM agent alone | 89% | 1,996 → 4,113 |
| Every tool in one prompt | 69% | 2,070 → 51,746 |
| Retrieval top 10 | 75% | 555 → 677 |

- The added vendors are distractors for the PayPal cases, which name no vendor; PayPal is
  set as the user's main account. With several vendors, Jev picks the service, then the
  resource within the likely services, then the operation. No question lists every tool.
- The Stripe, Square and GitHub catalogues are built from their public OpenAPI specs
  (pinned commits) by `build_openapi.py`. Their labels and ID links are inferred by rules,
  not reviewed by a person, and those vendors are measured at selection only.
- All 96 PayPal cases are counted at each size, including 20 marked seen. The 36
  new-vendor cases were written by the author before the run, and 3 of them are marked
  seen. The 16 runs shared rate limits in parallel, so their timings are not reported.
- Full report: `evals/results/scale-20261118-031812/summary.md`.

## How it works

Search narrows, structure decides, code executes.

1. **Catalogue at build time.** Each request becomes a resource plus an action. It
   also records the IDs it needs and returns, a risk tier, a body template and an
   example response. PayPal's labels were proposed by an LLM and reviewed by a person.
   The other vendors' labels are inferred by fixed rules from their OpenAPI specs.
2. **Selection.** Jev (TypeSafe's decision model) returns a probability for each option
   it is shown. It chooses the service (with several vendors), then the resource, then
   the operation. A tool's score is the product of these probabilities. Code then
   acts, asks on a near tie, or hands over to the LLM agent. The LLM agent takes the
   same steps, and every answer it gives is checked against the catalogue.
3. **Planning.** Code searches the ID graph backwards from the goal. "Send an invoice"
   becomes generate number, create draft, send. Existing items named in words are
   looked up.
4. **Several actions in one message.** A Jev gate decides whether a message holds
   several actions; if so, the LLM splits it. Earlier actions pass their IDs to later
   ones, and the user approves the whole set once.
5. **Filling.** The model fills slots in the request template. Unknown fields are
   rejected, values must come from the user's words, and IDs come from code. A
   readiness check asks for anything only the user can supply.
6. **Approval by risk tier.** Reads run straight away. Writes need a confirmation.
   Money, outbound and destructive calls need an explicit yes.
7. **Execution with evidence.** Each write is recorded before it is sent, with an
   idempotency key. After a crash, unfinished writes are reconciled with PayPal, never
   blindly resent.
8. **Compute and answer.** Totals, counts and filters are computed in code with exact
   decimals. The LLM only words the reply.

The conversation is a LangGraph state graph (`agent/agent.py`). It is checkpointed after
every node, and it pauses with interrupts for questions and approvals.

    split -> select -+- rag -> end
                     +- system_search -> end
                     +- ask -> await_answer -> select
                     +- plan -> lookup -> fill -> check_ready -+- await_detail -> apply_answer
                                                               +- await_approval -> execute
                                                               +- execute -> compute -> answer -> end

The two required tools:
- **RAG**: cited answers from the collection's documentation, using BM25.
- **System search**: what the agent can do, and the status of earlier requests.

## Setup

Python 3.10 or later.

    python -m venv .venv
    .venv\Scripts\activate            (macOS or Linux: source .venv/bin/activate)
    pip install -r requirements.txt

1. Copy `.env.example` to `.env`. Add one LLM key (OpenAI, Groq or Gemini) and, ideally,
   `JEV_API_KEY`. PayPal sandbox credentials are needed only for `--live`.
2. Export the PayPal collection from Postman (Collection v2.1) to
   `data/PayPal APIs.postman_collection.json`. It is git-ignored because it contains
   client credentials.
3. Run `python build_model.py` to rebuild `paypal_model.json`, which is already
   committed.

The commands below assume the virtual environment is active.

## Commands

    python chat.py                            chat; calls answered by a mock of the collection's examples
    python chat.py --live                     chat; calls go to the PayPal sandbox (sandbox URLs only)
    python chat.py --session NAME             continue a saved conversation
    python chat.py --no-jev                   the LLM agent chooses tools alone
    python chat.py --no-langsmith             keep traces local
    python demo.py "list my invoices"         show the selection and the plan only
    python traces.py last -v                  the latest turn as a step tree, with prompts
    python -m unittest discover -s tests      run the tests (no keys needed)

Eval (see `evals/README.md`):

    python evals/run_eval.py --level selection --no-cache          four variants at 110 tools
    python evals/run_eval.py --live --e2e-variants jev --no-cache  end to end on the sandbox
    python scripts/fetch_specs.py                                  download pinned Stripe, Square, GitHub specs
    python build_openapi.py                                        build their models
    python evals/run_scale.py --size 2220 --out evals/results/scale-x
    python evals/run_scale.py --report evals/results/scale-x

Sandbox helpers: `scripts/sandbox_smoke.py` runs read-only checks, and
`scripts/sandbox_seed.py` creates test data. `scripts/check_quota.py` shows which LLM
provider has quota left.

## Layout

    agent/agent.py          the LangGraph graph
    agent/select_jev.py     selection by Jev: service, resource, operation, thresholds, fallback
    agent/select.py         selection by the LLM agent, checked against the catalogue
    agent/split.py          several actions in one message
    agent/planner.py        backward search over the ID graph
    agent/resolve.py        find an existing item from a description
    agent/fill.py           slot filling, grounding, required fields
    agent/ready.py          readiness check and reading the user's answers
    agent/policy.py         approval per risk tier
    agent/execute.py        mock and sandbox transports, ID extraction
    agent/evidence.py       write-ahead record of every write
    agent/reconcile.py      settling unfinished writes after a crash
    agent/compute.py        totals, counts and filters in code
    agent/rag.py            BM25 retrieval and cited answers over knowledge/
    agent/system_search.py  capabilities and request status
    agent/catalog.py        the catalogue, one or several vendors
    agent/jev.py, llm.py    clients for Jev and the LLM providers
    agent/trace.py          spans per turn; LangSmith via agent/observability.py
    agent/config.py         every threshold, overridable by AGENT_* variables
    build_model.py          Postman export + reviewed labels -> paypal_model.json
    propose_review.py       LLM label proposals for a person to review
    build_openapi.py        OpenAPI spec -> model, by fixed rules
    evals/                  cases, scoring, runners, results
    tests/                  unit tests

## Known limits

- Jev's thresholds are starting values; they have not been tuned yet.
- Labels and ID links for Stripe, Square and GitHub are inferred by rules and not
  reviewed. Those vendors are measured at selection only, never called.
- The eval has 96 PayPal cases and 36 multi-vendor cases; more can be added in
  `evals/cases_user.jsonl`.
- A rejected API call (400) stops the run, with no automatic repair.
- Look-up covers 8 kinds of item, and reads the first 20 records.
- Lists paged by token (disputes) are computed over the first page only.
- RAG uses keyword search and covers only the collection's own descriptions (about 43 KB).
- LangSmith tracing sends prompts, responses and request bodies to LangSmith. Use
  `--no-langsmith` to keep traces local.
