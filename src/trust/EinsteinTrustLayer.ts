import type { Finding, ReasoningVerdict, Recommendation, TrustLayerResult } from "../types.js";

/**
 * Einstein Trust Layer (illustrative stand-in).
 *
 * Every reasoning-action output passes through this gate before it can reach a
 * reviewer or be written to the queue. Two checks, both must pass:
 *
 *   1. Schema validation — the output is exactly the expected shape.
 *   2. Grounding — the CVE the model says it reasoned about is the CVE we asked about.
 *
 * On any failure the verdict is replaced with a forced escalation. The model's
 * original confidence is discarded; a failed gate is never "mostly trusted".
 */

const RECOMMENDATIONS: readonly Recommendation[] = ["accept", "reject", "escalate"];
const EXPECTED_KEYS = ["confidence", "rationale", "recommendation", "referencedCveId"];

export interface SchemaCheck {
  ok: boolean;
  errors: string[];
  verdict?: ReasoningVerdict;
}

/** Validate that `raw` parses to exactly the ReasoningVerdict shape. */
export function validateSchema(raw: string): SchemaCheck {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { ok: false, errors: ["output is not valid JSON"] };
  }

  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { ok: false, errors: ["output is not a JSON object"] };
  }

  const obj = parsed as Record<string, unknown>;
  const errors: string[] = [];

  const keys = Object.keys(obj).sort();
  const missing = EXPECTED_KEYS.filter((k) => !keys.includes(k));
  const extra = keys.filter((k) => !EXPECTED_KEYS.includes(k));
  if (missing.length) errors.push(`missing field(s): ${missing.join(", ")}`);
  if (extra.length) errors.push(`unexpected field(s): ${extra.join(", ")}`);

  if (!RECOMMENDATIONS.includes(obj.recommendation as Recommendation)) {
    errors.push(`recommendation must be one of ${RECOMMENDATIONS.join("|")}`);
  }
  if (
    typeof obj.confidence !== "number" ||
    !Number.isFinite(obj.confidence) ||
    obj.confidence < 0 ||
    obj.confidence > 1
  ) {
    errors.push("confidence must be a number in [0, 1]");
  }
  if (typeof obj.rationale !== "string" || obj.rationale.trim() === "") {
    errors.push("rationale must be a non-empty string");
  }
  if (typeof obj.referencedCveId !== "string") {
    errors.push("referencedCveId must be a string");
  }

  if (errors.length) return { ok: false, errors };
  return { ok: true, errors: [], verdict: obj as unknown as ReasoningVerdict };
}

/** Exact-match grounding: the model must cite the finding's actual CVE ID. */
export function checkGrounding(verdict: ReasoningVerdict, finding: Finding): SchemaCheck {
  if (verdict.referencedCveId === finding.cveId) return { ok: true, errors: [] };
  return {
    ok: false,
    errors: [
      `grounding mismatch: model referenced "${verdict.referencedCveId}" but finding is "${finding.cveId}"`,
    ],
  };
}

export function forcedEscalation(finding: Finding, failures: string[]): ReasoningVerdict {
  return {
    recommendation: "escalate",
    confidence: 0,
    rationale: `Trust layer rejected model output: ${failures.join("; ")}`,
    referencedCveId: finding.cveId,
  };
}

export class EinsteinTrustLayer {
  evaluate(rawOutput: string, finding: Finding): TrustLayerResult {
    const schema = validateSchema(rawOutput);
    // Grounding needs a well-formed referencedCveId, so it can only run on a
    // schema-valid verdict. A schema failure is reported as a grounding failure too.
    const grounding = schema.verdict
      ? checkGrounding(schema.verdict, finding)
      : { ok: false, errors: ["grounding not evaluated: schema invalid"] };

    const failures = [...schema.errors, ...grounding.errors];
    const passed = schema.ok && grounding.ok;

    return {
      passed,
      schemaPassed: schema.ok,
      groundingPassed: grounding.ok,
      failures,
      verdict: passed && schema.verdict ? schema.verdict : forcedEscalation(finding, failures),
    };
  }
}
