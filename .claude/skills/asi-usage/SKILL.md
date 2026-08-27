---
description: Use the asi CLI: install/uninstall/sync/validate, catalog resolution, native `asi run` actions, machine-local model overrides, daemon/watch/schedule, and the render pipeline.
name: asi-usage
---

# ASI CLI Usage

## Purpose

Invoke this skill when an agent needs to operate the `asi` CLI in a repo that
consumes an asi catalog: install/update agent and skill assets, sync
render targets, run native skill actions, manage per-agent model overrides,
configure catalog resolution, and use the daemon/watch/schedule features.
This is the agent-facing guide to using `asi` — not how to extend it or
author new catalog content (see this catalog's authoring-oriented skills,
if any are installed, for that).

Concrete trigger examples:
1. Agent must install skills/agents into `.claude/`, `.github/`, `.codex/`, or `.gemini/`
2. Agent needs to run a native skill action via `asi run <skill-id>[:<action>]`
3. Agent needs to manage agent model pinning or a global model default
4. Agent needs to validate the catalog or inspect render output
5. Agent needs to sync catalog renders or manage schedules/watchers

## Prerequisites

- `asi` binary is on PATH (build from source: `go build ./cmd/asi`)
- Catalog resolution works (see Catalog Resolution below) — a bare clone
  with `catalog/agents.json` present at the repo root is enough
- When using `asi run`, verify the action is listed via `asi run --list`
  first — not every skill ships a native action

## Core Concepts

### Catalog → Targets Rendering

`asi` maintains a canonical catalog in `catalog/`:
- `catalog/agents.json` / `catalog/skills.json` — metadata registries
- `catalog/agents/{id}.md` — agent master bodies (shared across targets)
- `catalog/skills/{id}/SKILL.md` — master skill files
- `catalog/overrides/{target}/agents/` and `skills/` — target-specific frontmatter overrides
- `catalog/resolved/{target}/` — body-only target overrides (committed, so CI needs no LLM access)

Render pipeline priority per file:
1. Resolved body (`catalog/resolved/{target}/{type}/{filename}`) if it exists
2. Master body
3. Frontmatter — from overrides, falling back to `toolMappings` in `agents.json`

`targets/` is an optional, gitignored render cache. Default behavior is
in-memory rendering at install time — you don't need pre-generated files.

Render command:
```bash
go run ./cmd/asi sync --targets claude,copilot,openai,gemini
# or, once built:
asi sync --targets claude
```

### Catalog Resolution

`asi` locates its catalog in this order:
1. `catalog.dir` in `~/.asi/config.json`, if set — an explicit override
2. Remote sync (`catalog.source` = `remote`) — clones/pulls `remote.url` into a cache dir
3. `catalog/` beside the running binary (the shipped default for an installed binary)
4. Upward walk from CWD looking for `catalog/` — a dev-loop fallback (`go run` builds to a temp dir, so step 3 can't find the repo)
5. Hard error listing every location probed

Manage via:
```bash
asi config set-catalog-dir /path/to/repo    # parent of a catalog/ dir; pass "" to clear
asi config set-remote https://github.com/your-org/your-asi-catalog.git
asi config set-branch main
asi config set-catalog-source remote
asi config show
```

## CLI Commands

### Install / Uninstall

```bash
asi install --scope both --targets claude,copilot
asi install --non-interactive --scope local --targets claude --dry-run
asi uninstall --targets claude
asi status
```

Filtering flags (`install`/`uninstall`/`sync` all accept these, composed as
an intersection): `--category`, `--domain`, `--skill`, `--agent`.

Profiles:
```bash
asi install --scope profile:<name>
```
- Installs only a keep-listed subset of assets into a named profile directory
- Fails closed: a profile scope requires a keep-list and refuses a whole-catalog install
- No `--targets` needed for profile scope

Schedules:
```bash
asi install --install-schedules [--schedule-filter id1,id2]
```

### Validation & Sync

```bash
asi validate
asi sync --targets claude,copilot,openai,gemini
```

### Native Skill Actions

`asi run <skill-id>[:<action>]` invokes compiled Go actions under
`internal/skillrun/`. This is opt-in per skill — a skill with no
registered action is documentation-only, and `asi run` for it will fail.

List what's available:
```bash
asi run --list
```

A skill with a single action can be invoked as `asi run <skill-id>`; one
with several requires `<skill-id>:<action>`. Arguments after the selector
are forwarded to the action.

Note: actions are compiled into the binary, not sidecar scripts — a fork
that adds a native action registers it in `internal/skillrun/` and wires a
blank import into `internal/cli/run.go`; nothing external gets shelled out
to at runtime.

### Config & Model Overrides

Machine-local overrides live in `~/.asi/config.json` under `agent_models`.
They let one machine run an agent on a different model than the catalog
default, without touching the catalog.

Precedence for an agent's rendered `model:`:
0. `--agent-model id=model` / `--agent-model-all model` (ephemeral, this run only)
1a. `agent_models.agents.<id>` in config
1b. `agent_models.global` in config (skips agents listed in `agent_models.exclude`)
2. `catalog/overrides/{target}/agents/{id}.md` frontmatter
3. `modelHints.{target}` in `catalog/agents.json`
4. *(none)* — inherit whatever model the invoking session is already running on

Commands:
```bash
asi agent list-models
asi agent show-model <agent-id>
asi agent set-model <agent-id> <model>      # no args: interactive picker
asi agent unset-model <agent-id>
asi agent set-global-model <model>
asi config set-agent-model-exclude id1,id2
asi sync --targets claude --agent-model <agent-id>=<model>  # one-off, persists nothing
```

Don't assert facts about a model's capabilities or context window from
memory — `asi` doesn't validate model IDs against a live registry out of
the box (a fork may wire one up); `asi validate` only checks catalog
schema shape, and an unrecognized model ID there is a warning, not an
error, since the catalog is shared across tool ecosystems.

### Daemon, Watchers, Schedules

```bash
asi daemon status
asi watch list
asi watch create --from-catalog
asi schedule list
```

Schedules are defined in `catalog/schedules.json` and
`catalog/schedules/{id}.json` and installed via `install --install-schedules`.

### Other Shipped Commands

A few additional utility commands ship in the binary beyond the core
catalog/install/sync surface — `asi link-docs` (symlinks repo documentation
into a configured output directory, tunable via `asi config set-*` for
dev-root/vault-path/output-dir/skip-repos), `asi gateway` (manages AI
gateway configuration), and `asi update` (self-updates the binary and
catalog from the configured remote's latest release, where release
automation is wired up). Run `asi <command> --help` for the full flag set
of any of these before using them.

## Usage Patterns for Agents

### Installing Assets

When a user asks to install skills/agents:
1. Determine scope (`both`, `local`, `profile:<name>`) and targets (`claude`, `copilot`, `openai`, `gemini`)
2. Run `asi install --scope <scope> --targets <targets>` (add `--dry-run` first if unsure)
3. Report installed paths

### Running Native Actions

When a skill requires `asi run`:
1. Confirm the action exists: `asi run --list | grep <skill-id>`
2. Pass through parameters as documented in the skill's own `SKILL.md`
3. Capture output and relay results

### Validating Changes

After catalog edits:
1. `asi validate`
2. `asi sync --targets claude` (or whichever targets you're checking)
3. Verify rendered files under the `targets/` cache (optional — in-memory rendering doesn't require this step)
4. Never commit `targets/` — it's gitignored, generated on demand

### Model Pinning

If a user needs an agent on a different model:
1. Explain the machine-local override vs. catalog-default distinction
2. Use `asi agent set-model <id> <model>` for a local, non-committed override
3. For a catalog-wide change, edit `catalog/agents.json` `modelHints` or add a target override instead

## Hard Rules

1. Never commit `targets/` — it's a render cache, always regenerable from the catalog
2. Never write catalog changes without running `asi validate` first
3. `asi run` actions are compiled into the binary; do not invoke a sidecar script in its place
4. Model facts (capabilities, context windows) must come from a real, checked source — never assumed from memory
5. Catalog path resolution falls back to a CWD upward walk in dev (`go run`); an installed binary uses its sibling `catalog/` path instead
6. `asi install --scope profile:<name>` is fails-closed: it requires a keep-list and refuses a whole-catalog install

## Output Format

When reporting `asi` operations:
- **Command**: exact invocation used
- **Result**: success/failure with key output excerpts
- **Path**: where assets were installed/rendered
- **Validation**: `asi validate` status

## Anti-Patterns to Avoid

- Invoking `asi run` for a skill that has no native action — only documentation exists for most skills
- Modifying a `targets/` directory by hand — always regenerate from the catalog via `sync`
- Stating model capabilities or context windows from memory instead of a real source
- Assuming agents can read each other's bodies — a subagent only sees its own body plus its dispatch prompt, not the rest of the catalog
