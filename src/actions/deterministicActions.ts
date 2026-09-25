import type { DeterministicResult, Finding } from "../types.js";

/**
 * Deterministic actions: pure functions, no LLM, no I/O.
 *
 * Each rule either returns a conclusive disposition or declares itself
 * inconclusive. The subagent runs them in order and stops at the first
 * conclusive result.
 */

export const LOW_CVSS_THRESHOLD = 4.0;
export const LOW_EPSS_THRESHOLD = 0.1;
export const STALE_BASE_IMAGE_DAYS = 90;

const MS_PER_DAY = 24 * 60 * 60 * 1000;

/** CVSS < 4.0 AND EPSS < 0.1 → auto-close as accepted risk, low severity. */
export function checkSeverityThreshold(finding: Finding): DeterministicResult {
  if (finding.cvss < LOW_CVSS_THRESHOLD && finding.epss < LOW_EPSS_THRESHOLD) {
    return {
      conclusive: true,
      rule: "severity_threshold",
      disposition: "accepted_risk_low_severity",
      summary: `Accepted risk, low severity (CVSS ${finding.cvss} < ${LOW_CVSS_THRESHOLD}, EPSS ${finding.epss} < ${LOW_EPSS_THRESHOLD}).`,
    };
  }
  return {
    conclusive: false,
    rule: "severity_threshold",
    summary: `Above low-severity floor (CVSS ${finding.cvss}, EPSS ${finding.epss}).`,
  };
}

/** A fixed version exists and the image isn't on it → recommend rebuild. */
export function checkKnownFixAvailable(finding: Finding): DeterministicResult {
  if (finding.fixedVersion !== null && finding.installedVersion !== finding.fixedVersion) {
    return {
      conclusive: true,
      rule: "known_fix_available",
      disposition: "patch_available_recommend_rebuild",
      summary: `Patch available: ${finding.packageName} ${finding.installedVersion} → ${finding.fixedVersion}. Recommend rebuild.`,
    };
  }
  return {
    conclusive: false,
    rule: "known_fix_available",
    summary:
      finding.fixedVersion === null
        ? `No upstream fix for ${finding.packageName}.`
        : `Image already pinned to fixed version ${finding.fixedVersion}.`,
  };
}

/** Base image last rebuilt more than 90 days before `asOf` → stale. */
export function checkBaseImageFreshness(finding: Finding, asOf: Date): DeterministicResult {
  const rebuilt = new Date(finding.baseImageLastRebuilt);
  if (Number.isNaN(rebuilt.getTime())) {
    throw new Error(`Finding ${finding.id} has an invalid baseImageLastRebuilt date.`);
  }
  const ageDays = Math.floor((asOf.getTime() - rebuilt.getTime()) / MS_PER_DAY);
  if (ageDays > STALE_BASE_IMAGE_DAYS) {
    return {
      conclusive: true,
      rule: "base_image_freshness",
      disposition: "stale_base_image",
      summary: `Stale base image: last rebuilt ${ageDays} days ago (> ${STALE_BASE_IMAGE_DAYS}).`,
    };
  }
  return {
    conclusive: false,
    rule: "base_image_freshness",
    summary: `Base image fresh (${ageDays} days old).`,
  };
}

/**
 * Run every rule in priority order. Returns all evaluated results; the last
 * element is conclusive if any rule fired, otherwise every result is
 * inconclusive and the finding falls through to reasoning (or human handoff).
 */
export function runDeterministicActions(finding: Finding, asOf: Date): DeterministicResult[] {
  const rules = [
    () => checkSeverityThreshold(finding),
    () => checkKnownFixAvailable(finding),
    () => checkBaseImageFreshness(finding, asOf),
  ];
  const results: DeterministicResult[] = [];
  for (const rule of rules) {
    const result = rule();
    results.push(result);
    if (result.conclusive) break;
  }
  return results;
}
