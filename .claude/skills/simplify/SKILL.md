---
allowed-tools: Read Write Edit Bash Grep
description: Review the changed code for reuse, simplification, efficiency, and altitude cleanups, then apply the fixes. Quality only — it does not hunt for bugs.
name: simplify
---

# Simplify Skill

Automatically refactor code to be cleaner, more efficient, and more maintainable. Focuses on code quality improvements without fixing bugs.

## Trigger

Invoke this skill when you want to improve code quality after implementation:
- Before code review to eliminate low-hanging fruit
- To remove duplication and improve clarity
- To optimize performance-critical paths
- To extract reusable components or utilities

## Usage

```
/simplify
```

The skill:
1. Analyzes staged/uncommitted changes
2. Identifies redundancy, inefficiency, and clarity improvements
3. Applies fixes directly to the working tree
4. Stages the refactored code for your review

## What Gets Simplified

1. **Reuse** — extract common patterns into helpers or libraries
2. **Clarity** — rename variables, extract methods, improve structure
3. **Efficiency** — remove unnecessary allocations, cache lookups, improve algorithms
4. **Altitude** — remove obvious dead code, collapse single-use constructs
5. **Dependencies** — reduce imports, eliminate transitive coupling

## What Simplify Does NOT Do

- **Bug fixes** — correctness issues stay as-is (use code-review for those)
- **Refactoring** — large structural changes (use refactoring-specialist for those)
- **Tests** — test quality or coverage (handle separately)
- **Backwards-compatibility** — assumes you're not shipping this API yet

## Anti-Patterns

- Don't simplify code you don't understand
- Don't apply all suggestions blindly — review the diffs
- Don't use this on shipping code without explicit intent to refactor
- Avoid simplifying for the sake of being clever (readability > cleverness)

## Example

```bash
git add .
/simplify
```

Refactors the staged changes and leaves them in the working tree for review.
