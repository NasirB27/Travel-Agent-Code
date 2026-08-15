# DC Rental Property — AI Automation

An AI-assisted tenant-screening and property-management pipeline for a
4-unit DC property, built around a [LangGraph](https://github.com/langchain-ai/langgraph)
state machine with human approval required before every consequential
action (rejecting an applicant, sending a lease, publishing an ad, sending
a legal notice).

See **[docs/architecture.md](docs/architecture.md)** for the full
architecture, component evaluations, compliance notes, and roadmap.

## Status

Phase 1 of the roadmap is implemented: an intake → document-parsing →
screening-score → human-approval → lease-draft → human-approval pipeline,
runnable end to end from the command line. Everything else in
`docs/architecture.md` (lead triage, ad automation, the concierge chatbot,
e-sign) is still ahead.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py
```

`main.py` runs one demo applicant through the graph and prompts you in the
terminal at each human-in-the-loop checkpoint — this is standing in for the
Slack/SMS approval channel described in the architecture doc; the graph
itself doesn't care how the human response arrives.

Set `ANTHROPIC_API_KEY` to have the application parser extract financials
with Claude instead of the offline fallback parser (see
`graph/nodes/application_parser.py`).

Every human decision is appended to `audit_log.jsonl` (gitignored).

## Tests

```bash
pytest
```

`tests/test_graph.py` drives the graph through both interrupt points
(screening decision, lease-send approval) and both outcomes (approved,
rejected) without needing terminal input or network access.

## Repo structure

```
graph/
  state.py               # ApplicantState schema
  audit.py                # append-only decision log
  build.py                # wires the StateGraph together
  nodes/
    intake.py
    application_parser.py # document -> financials (Claude, or offline fallback)
    screening.py           # fixed-criteria pass/fail scoring
    approval.py             # the two human-in-the-loop interrupts
    lease.py                 # lease draft + rejection logging
knowledge_base/
  property_facts.md       # seed for the future concierge chatbot's RAG source
docs/
  architecture.md         # full architecture plan and roadmap
main.py                    # CLI demo entrypoint
tests/
  test_graph.py
```
