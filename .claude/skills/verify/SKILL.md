---
allowed-tools: Bash Read PowerShell
description: Verify that a code change actually does what it's supposed to by running the app and observing behavior.
name: verify
---

# Verify Skill

Test a code change by running the application and observing its behavior, confirming the change works as intended.

## Trigger

Invoke this skill when you want to manually verify that a code change works:
- After implementing a new feature to confirm it behaves correctly
- To test edge cases and user interactions before committing
- To catch regressions in related functionality
- To validate that a bug fix actually resolves the issue

## Usage

```
/verify
```

The skill automatically:
1. Detects the project type (CLI, server, frontend, library, TUI, etc.)
2. Starts/builds the application if needed
3. Guides you through manual testing of the golden path
4. Prompts for testing edge cases
5. Watches for regressions in related features

## What Gets Tested

- **Golden path** — the primary happy-path workflow
- **Edge cases** — boundary conditions, empty inputs, max values
- **Error handling** — invalid inputs, permission errors, timeout handling
- **Related features** — check that nearby functionality still works
- **Performance** — basic responsiveness checks (not benchmarks)

## Anti-Patterns

- Don't rely on verify alone (combine with unit tests and code review)
- Don't skip manual testing because CI is green (type-checking ≠ behavior)
- Don't test incomplete implementations (they'll be obviously broken)
- Avoid testing the same change twice without making new changes

## Limitations

- Cannot test UI changes without a running browser or terminal
- Cannot test background jobs or scheduled tasks without waiting
- Cannot verify distributed system behavior in isolation

Report these limitations explicitly rather than claiming success without evidence.

## Example: Frontend Change

```bash
/verify
```

Starts the dev server, opens the browser, and prompts:
- "Test the new button you added"
- "What happens when you click it with invalid input?"
- "Does the loading state appear?"
- "Check the sidebar still scrolls correctly"
