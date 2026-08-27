---
allowed-tools: Read Bash Grep Glob
description: Review the current diff for correctness bugs and reuse/simplification/efficiency cleanups at the given effort level (low/medium: fewer, high-confidence findings; high/max: broader coverage, may include uncertain findings).
name: code-review
---

# Code Review Skill

Review the current git diff for correctness bugs and code quality improvements.

## Trigger

Invoke this skill when you want an independent review of uncommitted or staged changes:
- Before committing or opening a PR to catch bugs early
- To get a second opinion on code quality and simplification opportunities
- To verify security and performance implications of changes

## Usage

```
/code-review [--low|--medium|--high|--max] [--fix|--comment]
```

### Effort Levels

Effort controls **how deep the investigation goes** — how many files are read, how far call chains are traced, how many edge cases are considered. It does **not** decide what gets reported; that is a separate step (see Filtering Stage below).

- **--low** — Quick scan; investigate the diff itself and immediate call sites only (fastest, most conservative)
- **--medium** — Standard review; investigate the diff plus direct callers/callees (default)
- **--high** — Thorough review; trace transitive call chains and cross-file interactions
- **--max** — Exhaustive analysis; full blast-radius investigation including edge cases and design implications
- **ultra** — Deep multi-agent cloud review with independent perspectives, each at `--max` depth

### Output Modes

- **Default** — Report findings to stdout
- **--fix** — Apply findings to the working tree after review
- **--comment** — Post findings as inline PR comments (requires PR context)

## Finding Stage

Investigate at the depth implied by the effort level above, but **do not filter while investigating**. Enumerate every finding you notice — including ones you're unsure about — as a candidate. For each candidate, record:

- **Description** — what the issue is and where (file:line)
- **severity** — `high` (breaks correctness/security in realistic paths), `medium` (real but bounded impact — perf, maintainability, edge-case correctness), or `low` (style, minor simplification, speculative)
- **confidence** — `high` (verified against the actual code/call chain), `medium` (plausible, reasoning has a gap), or `low` (a hunch worth surfacing but not fully traced)

This produces a complete candidate list, independent of effort level. A `--low` run and a `--max` run investigating the same diff to the same depth should enumerate the same candidates — they differ only in how much ground was covered before enumerating (per the depth rules above), never in how aggressively the list gets pruned.

## Filtering Stage

After the Finding Stage produces the full candidate list, apply the effort flag's **reporting threshold** as a separate pass. This is a filter over already-completed findings, not a re-investigation:

| Effort | Keep findings where |
|---|---|
| `--low` | `severity: high` AND `confidence: high` |
| `--medium` | `severity: high or medium` AND `confidence: high or medium` |
| `--high` | `severity: high or medium` (any confidence), plus `severity: low` AND `confidence: high` |
| `--max` / `ultra` | everything (no filtering) |

State findings dropped by this pass are not lost silently mid-review — they were fully investigated and reasoned through, then excluded only at the final filter. Never fold the threshold back into the investigation step: "only report high-confidence findings" is an instruction about what to print, not an instruction to look less carefully or stop enumerating early.

## Dimensions Covered

1. **Correctness** — off-by-one errors, null dereference, logic bugs
2. **Performance** — inefficient loops, unnecessary allocations, cache misses
3. **Security** — input validation, secrets in logs, privilege boundaries
4. **Maintainability** — reusability, naming clarity, DRY violations
5. **Tests** — coverage, assertion quality, test isolation

## Anti-Patterns

- Never change code without understanding why the change is correct
- Don't blindly accept findings without reasoning through them
- Avoid reviewing code that hasn't been staged (use `git add` first)
- Don't use this as a substitute for automated checks (linters, type checkers)
- Don't let the Filtering Stage's threshold leak back into the Finding Stage — investigate at full effort-implied depth first, filter second, and never skip enumerating a finding because you expect it to be filtered out

## Example: Standard Review

```bash
git add .
/code-review --high
```

Reports bugs and simplifications for the staged changes.

## Example: Fix Mode

```bash
git add .
/code-review --fix
```

After reporting findings, automatically applies corrections to the working tree.
