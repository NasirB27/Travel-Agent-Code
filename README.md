# DC Rental Property — AI Automation

An AI-assisted tenant-screening and property-management pipeline for a
4-unit DC property, built around a [LangGraph](https://github.com/langchain-ai/langgraph)
state machine with human approval required before every consequential
action (rejecting an applicant, sending a lease, publishing an ad, sending
a legal notice).

See **[docs/architecture.md](docs/architecture.md)** for the full
architecture, component evaluations, compliance notes, and roadmap.

## Status

Roadmap steps 1–3 from `docs/architecture.md` are implemented:

- **Applicant screening** (`graph/build.py`): intake → document parsing →
  screening score → human approval → lease draft → human approval.
- **Lead triage** (`graph/lead_build.py`): intake → classify into
  QUALIFIED / FOLLOW_UP / SUPPORT / SPAM → draft reply → human approval
  (SPAM skips review — discarding junk isn't a consequential decision).

Both are runnable end to end from the command line. Everything else in
`docs/architecture.md` (ad automation, the concierge chatbot, e-sign, and
wiring a QUALIFIED lead automatically into the applicant graph) is still
ahead.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py          # applicant-screening demo
python triage_lead.py   # lead-triage demo
```

Both scripts prompt you in the terminal at each human-in-the-loop
checkpoint — standing in for the Slack/SMS approval channel described in
the architecture doc; the graphs themselves don't care how the human
response arrives.

Set `ANTHROPIC_API_KEY` to have the application parser and lead
classifier/reply-drafter use Claude instead of their offline fallbacks
(see `graph/nodes/application_parser.py` and `graph/nodes/lead_triage.py`).

Every human decision is appended to `audit_log.jsonl` (gitignored).

## Tests

```bash
pytest
```

- `tests/test_graph.py` drives the applicant graph through both interrupt
  points and both outcomes (approved, rejected).
- `tests/test_lead_graph.py` drives the lead-triage graph through all four
  categories, the SPAM short-circuit, and both reply-approval outcomes
  (including a human edit).

Neither suite needs terminal input or network access.

## Repo structure

```
graph/
  state.py                # ApplicantState schema
  lead_state.py            # LeadState schema
  audit.py                  # append-only decision log, shared across pipelines
  build.py                   # wires the applicant-screening StateGraph
  lead_build.py                # wires the lead-triage StateGraph
  nodes/
    intake.py
    application_parser.py    # document -> financials (Claude, or offline fallback)
    screening.py               # fixed-criteria pass/fail scoring
    approval.py                  # the applicant graph's human-in-the-loop interrupts
    lease.py                       # lease draft + rejection logging
    lead_intake.py
    lead_triage.py                  # DM/email -> QUALIFIED/FOLLOW_UP/SUPPORT/SPAM
    lead_reply.py                     # drafts a reply + human approval interrupt
    lead_routing.py                     # SPAM short-circuit logging
knowledge_base/
  property_facts.md       # seed for the future concierge chatbot's RAG source
docs/
  architecture.md         # full architecture plan and roadmap
main.py                    # applicant-screening CLI demo
triage_lead.py               # lead-triage CLI demo
tests/
  test_graph.py
  test_lead_graph.py
```
