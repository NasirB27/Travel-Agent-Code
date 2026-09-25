import { describe, expect, it } from "vitest";
import { EinsteinTrustLayer, validateSchema } from "../src/trust/EinsteinTrustLayer.js";
import { makeFinding, verdictJson } from "./helpers.js";

const trust = new EinsteinTrustLayer();
const finding = makeFinding({ cveId: "CVE-2099-90001" });

describe("EinsteinTrustLayer — passing output", () => {
  it("passes a well-formed, grounded verdict through unchanged", () => {
    const r = trust.evaluate(verdictJson({ referencedCveId: finding.cveId, confidence: 0.83 }), finding);
    expect(r.passed).toBe(true);
    expect(r.failures).toEqual([]);
    expect(r.verdict).toMatchObject({ recommendation: "accept", confidence: 0.83 });
  });
});

describe("EinsteinTrustLayer — grounding check", () => {
  it("fails a mismatched CVE ID (hallucination) and forces escalation", () => {
    const r = trust.evaluate(
      verdictJson({ referencedCveId: "CVE-2099-99999", recommendation: "accept", confidence: 0.99 }),
      finding,
    );
    expect(r.passed).toBe(false);
    expect(r.schemaPassed).toBe(true);
    expect(r.groundingPassed).toBe(false);
    expect(r.failures.join()).toMatch(/grounding mismatch/);
    // Original high confidence is discarded.
    expect(r.verdict).toMatchObject({ recommendation: "escalate", confidence: 0, referencedCveId: finding.cveId });
  });

  it("requires an exact match (no case or whitespace tolerance)", () => {
    const r = trust.evaluate(verdictJson({ referencedCveId: " cve-2099-90001" }), finding);
    expect(r.groundingPassed).toBe(false);
  });
});

describe("EinsteinTrustLayer — schema validation", () => {
  const malformed: Array<[string, string]> = [
    ["non-JSON text", "Sure! I think you should accept this."],
    ["a JSON array", "[]"],
    ["an unknown recommendation", verdictJson({ referencedCveId: finding.cveId, recommendation: "approve" as never })],
    ["confidence above 1", verdictJson({ referencedCveId: finding.cveId, confidence: 1.7 })],
    ["negative confidence", verdictJson({ referencedCveId: finding.cveId, confidence: -0.1 })],
    ["confidence as a string", verdictJson({ referencedCveId: finding.cveId, confidence: "0.9" as never })],
    ["an empty rationale", verdictJson({ referencedCveId: finding.cveId, rationale: "   " })],
    ["a missing field", JSON.stringify({ recommendation: "accept", confidence: 0.9, referencedCveId: finding.cveId })],
    ["an extra field", JSON.stringify({ ...JSON.parse(verdictJson({ referencedCveId: finding.cveId })), severity: "Low" })],
    ["empty output (refusal or transport error)", ""],
  ];

  it.each(malformed)("fails %s and forces escalation", (_label, raw) => {
    const r = trust.evaluate(raw, finding);
    expect(r.passed).toBe(false);
    expect(r.schemaPassed).toBe(false);
    expect(r.verdict.recommendation).toBe("escalate");
    expect(r.verdict.confidence).toBe(0);
  });

  it("accepts the boundary confidences 0 and 1", () => {
    expect(validateSchema(verdictJson({ referencedCveId: "x", confidence: 0 })).ok).toBe(true);
    expect(validateSchema(verdictJson({ referencedCveId: "x", confidence: 1 })).ok).toBe(true);
  });
});
