import { describe, expect, it } from "vitest";
import { parseIlLevel, readIlLevelFromEnv, reasoningAllowed } from "../src/config/ilLevel.js";
import { createTriageSystem } from "../src/system.js";
import { groundedLlm, makeFinding, makeStore } from "./helpers.js";

describe("IL level config", () => {
  it("defaults to IL4 when IL_LEVEL is unset or empty", () => {
    expect(readIlLevelFromEnv({})).toBe("IL4");
    expect(parseIlLevel("")).toBe("IL4");
  });

  it("parses valid values case-insensitively", () => {
    expect(parseIlLevel("il5")).toBe("IL5");
    expect(parseIlLevel(" IL2 ")).toBe("IL2");
  });

  it("throws on invalid values rather than silently enabling reasoning", () => {
    expect(() => parseIlLevel("IL-5")).toThrow(/Invalid IL_LEVEL/);
    expect(() => parseIlLevel("IL6")).toThrow();
  });

  it("allows reasoning at IL2 and IL4 only", () => {
    expect(reasoningAllowed("IL2")).toBe(true);
    expect(reasoningAllowed("IL4")).toBe(true);
    expect(reasoningAllowed("IL5")).toBe(false);
  });
});

describe("IL5 behavior", () => {
  const inconclusive = [
    makeFinding({ id: "A", cveId: "CVE-2099-90001", severity: "Medium" }),
    makeFinding({ id: "B", cveId: "CVE-2099-90002", severity: "High" }),
    makeFinding({ id: "C", cveId: "CVE-2099-90003", severity: "Critical" }),
  ];
  const deterministic = [
    makeFinding({ id: "D", cveId: "CVE-2099-90004", cvss: 2.0, epss: 0.01, severity: "Low" }),
    makeFinding({ id: "E", cveId: "CVE-2099-90005", fixedVersion: "2.0.0" }),
  ];

  it("never invokes the LLM, and every inconclusive finding lands in human handoff as il5_mode", async () => {
    const { client, complete } = groundedLlm();
    const system = createTriageSystem({
      findingsStore: makeStore([...inconclusive, ...deterministic]),
      llm: client,
      ilLevel: "IL5",
    });

    const outcomes = await system.engine.triageAll();

    expect(complete).not.toHaveBeenCalled();
    const byId = new Map(outcomes.map((o) => [o.finding.id, o]));
    for (const f of inconclusive) {
      expect(byId.get(f.id)?.subagentResult.kind).toBe("reasoning_not_permitted");
      expect(byId.get(f.id)?.handoffReason).toBe("il5_mode");
    }
    const pending = system.handoffQueue.list({ status: "pending" });
    expect(pending.map((i) => i.findingId).sort()).toEqual(["A", "B", "C"]);
    expect(pending.every((i) => i.reason === "il5_mode" && i.ilLevel === "IL5")).toBe(true);
  });

  it("still lets deterministic rules auto-resolve (degradation, not shutdown)", async () => {
    const { client } = groundedLlm();
    const system = createTriageSystem({ findingsStore: makeStore(deterministic), llm: client, ilLevel: "IL5" });

    const outcomes = await system.engine.triageAll();

    expect(outcomes.map((o) => o.decision.outcome)).toEqual(["auto_closed_accepted_risk", "flagged_patch_available"]);
    expect(outcomes.every((o) => o.handoffReason === null)).toBe(true);
  });

  it("the same inconclusive findings DO reach the LLM at IL4 and IL2", async () => {
    for (const ilLevel of ["IL2", "IL4"] as const) {
      const { client, complete } = groundedLlm();
      const system = createTriageSystem({ findingsStore: makeStore(inconclusive), llm: client, ilLevel });
      await system.engine.triageAll();
      expect(complete).toHaveBeenCalledTimes(inconclusive.length);
    }
  });
});
