# Project notes for Claude

- Use the TypeSafe skill (`typesafe@typesafe-ai` plugin, enabled in `.claude/settings.json`) when working on this project — especially wherever a feature needs semantic judgment (lead triage, routing, ranking, extraction, verification) that could be a typed TypeSafe/Jev decision instead of a prompt-and-parse LLM step.
- The TypeSafe docs at https://docs.typesafe.ai/llms.txt are the source of truth for API/SDK details. Keep API keys server-side.
