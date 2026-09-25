import type { LlmClient } from "../llm/LlmClient.js";
import type { Finding } from "../types.js";

/**
 * Reasoning actions: the only place the system calls an LLM.
 *
 * This module builds the prompt, calls the model, and returns the model's raw
 * output. It intentionally does NOT parse, trust, or act on that output — the
 * raw text goes straight to the Einstein Trust Layer.
 */

export const VERDICT_JSON_SCHEMA: Record<string, unknown> = {
  type: "object",
  properties: {
    recommendation: { type: "string", enum: ["accept", "reject", "escalate"] },
    confidence: { type: "number" },
    rationale: { type: "string" },
    referencedCveId: { type: "string" },
  },
  required: ["recommendation", "confidence", "rationale", "referencedCveId"],
  additionalProperties: false,
};

export const REASONING_SYSTEM_PROMPT = [
  "You review vulnerability justifications for hardened container images.",
  "A maintainer has submitted a written justification for why a CVE finding should be accepted",
  "rather than remediated. Judge whether that justification adequately addresses this specific",
  "CVE's actual risk in this specific image.",
  "",
  "Recommend:",
  '- "accept" if the justification is specific, technically credible, and addresses the actual exploit path.',
  '- "reject" if it is generic, boilerplate, or does not engage with this CVE.',
  '- "escalate" if you cannot tell from the information given.',
  "",
  "Set confidence between 0 and 1. Echo the finding's CVE ID exactly in referencedCveId.",
  "Respond with JSON only.",
].join("\n");

export function buildJustificationReviewPrompt(finding: Finding): string {
  return [
    "Finding under review:",
    `- CVE ID: ${finding.cveId}`,
    `- Package: ${finding.packageName} ${finding.installedVersion}`,
    `- Fixed version: ${finding.fixedVersion ?? "none available"}`,
    `- Severity: ${finding.severity}`,
    `- CVSS: ${finding.cvss}`,
    `- EPSS: ${finding.epss}`,
    `- Image: ${finding.imageName}:${finding.imageTag}`,
    "",
    "Submitted justification:",
    '"""',
    finding.justification,
    '"""',
    "",
    "Does this justification adequately address this CVE's actual risk?",
  ].join("\n");
}

/**
 * Ask the model to judge the submitted justification. Returns raw model text.
 * A transport failure is reported as empty output, which the trust layer will
 * fail closed — an outage can never produce an automated decision.
 */
export async function reviewJustification(
  finding: Finding,
  llm: LlmClient,
): Promise<{ rawOutput: string; error?: string }> {
  try {
    const rawOutput = await llm.complete({
      system: REASONING_SYSTEM_PROMPT,
      prompt: buildJustificationReviewPrompt(finding),
      jsonSchema: VERDICT_JSON_SCHEMA,
    });
    return { rawOutput };
  } catch (err) {
    return { rawOutput: "", error: err instanceof Error ? err.message : String(err) };
  }
}
