import { describe, expect, it } from "vitest";
import { resolve } from "node:path";
import { MockFindingsStore } from "../src/data/mockFindingsStore.js";
import { ReplayLlmClient } from "../src/llm/LlmClient.js";
import { createTriageSystem } from "../src/system.js";
import type { Finding } from "../src/types.js";
import { groundedLlm, makeFinding, makeStore, mockLlm, verdictJson } from "./helpers.js";

function systemFor(finding: Finding, llm = groundedLlm()) {
  const system = createTriageSystem({ findingsStore: makeStore([finding]), llm: llm.client, ilLevel: "IL4" });
  return { ...system, complete: llm.complete };
}

describe("AtlasReasoningEngine — happy paths", () => {
  it("deterministic: auto-closes a low-severity finding with no LLM call and one audit entry", async () => {
    const { engine, queueApi, handoffQueue, complete } = systemFor(makeFinding({ cvss: 2.1, epss: 0.01, severity: "Low" }));

    const outcome = await engine.triage("T-001");

    expect(complete).not.toHaveBeenCalled();
    expect(outcome.decision.outcome).toBe("auto_closed_accepted_risk");
    expect(outcome.source).toBe("deterministic");
    expect(handoffQueue.list()).toHaveLength(0);
    expect(queueApi.getAuditLog()).toMatchObject([
      { findingId: "T-001", source: "deterministic", ilLevel: "IL4", trustLayerPassed: null },
    ]);
  });

  it("reasoning: a grounded, confident accept is written as an automated decision", async () => {
    const { engine, queueApi, handoffQueue, complete } = systemFor(makeFinding());

    const outcome = await engine.triage("T-001");

    expect(complete).toHaveBeenCalledTimes(1);
    expect(outcome.handoffReason).toBeNull();
    expect(outcome.decision).toMatchObject({ outcome: "justification_accepted", confidence: 0.9 });
    expect(handoffQueue.list()).toHaveLength(0);
    expect(queueApi.getAuditLog()[0]).toMatchObject({ source: "reasoning", trustLayerPassed: true });
    expect(outcome.trace.map((s) => s.node)).toEqual([
      "AtlasReasoningEngine",
      "deterministicActions",
      "deterministicActions",
      "deterministicActions",
      "TriageSubagent",
      "reasoningActions",
      "EinsteinTrustLayer",
      "mockCveQueueApi",
    ]);
  });

  it("reasoning: a confident reject is also automated", async () => {
    const { engine } = systemFor(makeFinding(), groundedLlm({ recommendation: "reject", confidence: 0.95 }));
    expect((await engine.triage("T-001")).decision.outcome).toBe("justification_rejected");
  });
});

describe("AtlasReasoningEngine — escalation paths", () => {
  it("trust layer failure: a hallucinated CVE ID routes to human handoff despite 0.99 confidence", async () => {
    const llm = mockLlm(() => verdictJson({ referencedCveId: "CVE-2099-00000", confidence: 0.99 }));
    const { engine, queueApi, handoffQueue } = systemFor(makeFinding(), llm);

    const outcome = await engine.triage("T-001");

    expect(outcome.handoffReason).toBe("trust_layer_failed");
    expect(outcome.decision).toMatchObject({ outcome: "pending_human_review", recommendation: "escalate", confidence: 0 });
    expect(handoffQueue.list({ status: "pending" })).toMatchObject([{ findingId: "T-001", reason: "trust_layer_failed" }]);
    expect(queueApi.getAuditLog()[0]).toMatchObject({ trustLayerPassed: false });
  });

  it("trust layer failure: an LLM outage fails closed into human handoff", async () => {
    const llm = mockLlm(() => {
      throw new Error("503");
    });
    const { engine } = systemFor(makeFinding(), llm);
    expect((await engine.triage("T-001")).handoffReason).toBe("trust_layer_failed");
  });

  it("low confidence: below threshold routes to human", async () => {
    const { engine } = systemFor(makeFinding(), groundedLlm({ confidence: 0.6 }));
    expect((await engine.triage("T-001")).handoffReason).toBe("low_confidence");
  });

  it("model escalation: an 'escalate' recommendation routes to human", async () => {
    const { engine } = systemFor(makeFinding(), groundedLlm({ recommendation: "escalate", confidence: 0.95 }));
    expect((await engine.triage("T-001")).handoffReason).toBe("model_recommended_escalation");
  });

  it("critical severity: routes to human even with a clean, confident model accept", async () => {
    const { engine } = systemFor(makeFinding({ severity: "Critical" }));
    expect((await engine.triage("T-001")).handoffReason).toBe("critical_severity");
  });

  it("critical severity: routes to human even when a deterministic rule fired", async () => {
    const { engine, complete } = systemFor(makeFinding({ severity: "Critical", fixedVersion: "9.9.9" }));
    const outcome = await engine.triage("T-001");
    expect(complete).not.toHaveBeenCalled();
    expect(outcome.handoffReason).toBe("critical_severity");
    expect(outcome.decision.detail).toMatch(/Patch available/);
  });

  it("a reviewer resolving the item writes a human-sourced decision at the original IL", async () => {
    const { engine, queueApi, handoffQueue } = systemFor(makeFinding(), groundedLlm({ confidence: 0.4 }));
    await engine.triage("T-001");

    handoffQueue.resolve("T-001", "reject", "Justification lacks runtime evidence.");

    expect(handoffQueue.list({ status: "pending" })).toHaveLength(0);
    expect(queueApi.getAuditLog().map((e) => e.source)).toEqual(["reasoning", "human"]);
    expect(queueApi.getAuditLog()[1]).toMatchObject({
      ilLevel: "IL4",
      decision: { outcome: "justification_rejected", handoffReason: "low_confidence" },
    });
  });
});

describe("AtlasReasoningEngine — bundled demo dataset", () => {
  const root = resolve(import.meta.dirname, "..");
  const run = (ilLevel: "IL4" | "IL5") =>
    createTriageSystem({
      findingsStore: MockFindingsStore.fromFile(resolve(root, "mock-data/findings.json")),
      llm: ReplayLlmClient.fromFile(resolve(root, "mock-data/recorded-llm-responses.json")),
      ilLevel,
    }).engine.triageAll();

  it("exercises every handoff reason at IL4 and collapses to il5_mode at IL5", async () => {
    const reasons = (outcomes: Awaited<ReturnType<typeof run>>) =>
      new Set(outcomes.map((o) => o.handoffReason).filter(Boolean));

    expect(reasons(await run("IL4"))).toEqual(
      new Set(["critical_severity", "model_recommended_escalation", "low_confidence", "trust_layer_failed"]),
    );
    expect(reasons(await run("IL5"))).toEqual(new Set(["critical_severity", "il5_mode"]));
  });
});
