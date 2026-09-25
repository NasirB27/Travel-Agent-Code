import { vi } from "vitest";
import { MockFindingsStore } from "../src/data/mockFindingsStore.js";
import type { LlmClient } from "../src/llm/LlmClient.js";
import type { Finding, ReasoningVerdict } from "../src/types.js";

export const SNAPSHOT_DATE = "2026-09-15";
export const AS_OF = new Date(SNAPSHOT_DATE);

/** A finding that no deterministic rule can settle, so it reaches reasoning. */
export function makeFinding(overrides: Partial<Finding> = {}): Finding {
  return {
    id: "T-001",
    cveId: "CVE-2099-90001",
    packageName: "libexample",
    installedVersion: "1.0.0",
    fixedVersion: null,
    severity: "High",
    cvss: 7.5,
    epss: 0.2,
    imageName: "demo-registry.local/hardened/test",
    imageTag: "1.0",
    baseImageLastRebuilt: "2026-09-01",
    justification: "The vulnerable function is never called; verified by tracing.",
    ...overrides,
  };
}

export function makeStore(findings: Finding[]): MockFindingsStore {
  return new MockFindingsStore({ snapshotDate: SNAPSHOT_DATE, findings });
}

export function verdictJson(overrides: Partial<ReasoningVerdict> & { referencedCveId: string }): string {
  return JSON.stringify({
    recommendation: "accept",
    confidence: 0.9,
    rationale: "Specific, evidence-backed justification.",
    ...overrides,
  });
}

/**
 * Mock LlmClient. `respond` maps each request to raw model text; the spy lets
 * tests assert exactly how many times the model was (or was not) called.
 */
export function mockLlm(respond: (prompt: string) => string | Promise<string>) {
  const complete = vi.fn(async (req: { prompt: string }) => respond(req.prompt));
  const client: LlmClient = { name: "mock-llm", complete };
  return { client, complete };
}

/** Mock that grounds correctly on whatever CVE is in the prompt. */
export function groundedLlm(overrides: Partial<ReasoningVerdict> = {}) {
  return mockLlm((prompt) => {
    const cveId = prompt.match(/CVE-\d{4}-\d+/)?.[0] ?? "";
    return verdictJson({ referencedCveId: cveId, ...overrides });
  });
}
