import type { IlLevel } from "../types.js";

/**
 * Impact Level configuration.
 *
 * DoD Impact Levels govern which services — including which AI models — are
 * authorized to process a given class of data. This module is the ONLY place
 * that encodes what each level permits. Nothing else in the codebase should
 * compare against a specific IL string.
 */

export const IL_LEVELS = ["IL2", "IL4", "IL5"] as const satisfies readonly IlLevel[];

export const DEFAULT_IL_LEVEL: IlLevel = "IL4";

export function isIlLevel(value: string): value is IlLevel {
  return (IL_LEVELS as readonly string[]).includes(value);
}

/**
 * Parse an IL level from a raw string (typically the IL_LEVEL env var).
 * Missing/empty → default. Anything else invalid → throw, so a typo like
 * "IL-5" can never silently run with reasoning enabled.
 */
export function parseIlLevel(raw: string | undefined): IlLevel {
  const value = raw?.trim().toUpperCase();
  if (!value) return DEFAULT_IL_LEVEL;
  if (!isIlLevel(value)) {
    throw new Error(`Invalid IL_LEVEL "${raw}". Expected one of: ${IL_LEVELS.join(", ")}.`);
  }
  return value;
}

export function readIlLevelFromEnv(env: NodeJS.ProcessEnv = process.env): IlLevel {
  return parseIlLevel(env.IL_LEVEL);
}

/**
 * Whether an LLM-backed reasoning action may run at this Impact Level.
 *
 * In this model, IL5 workloads have no authorized LLM, so reasoning is off and
 * anything the deterministic rules can't settle goes to a human. This is the
 * single source of truth for that constraint.
 */
export function reasoningAllowed(ilLevel: IlLevel): boolean {
  return ilLevel !== "IL5";
}
