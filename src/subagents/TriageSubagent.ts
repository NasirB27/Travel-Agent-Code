import { runDeterministicActions } from "../actions/deterministicActions.js";
import { reviewJustification } from "../actions/reasoningActions.js";
import { reasoningAllowed } from "../config/ilLevel.js";
import type { LlmClient } from "../llm/LlmClient.js";
import type { EinsteinTrustLayer } from "../trust/EinsteinTrustLayer.js";
import type { Finding, IlLevel, SubagentResult, TraceStep } from "../types.js";

/**
 * TriageSubagent: the specialist the orchestrator routes CVE findings to.
 *
 * Order of operations:
 *   1. Deterministic actions. Conclusive → done, no LLM call.
 *   2. Impact Level gate. Reasoning not permitted → hand back for human handoff.
 *   3. Reasoning action → Einstein Trust Layer → gated verdict.
 *
 * The subagent never writes decisions or enqueues handoffs itself; it returns a
 * typed result and the orchestrator owns the final routing.
 */
export class TriageSubagent {
  constructor(
    private readonly llm: LlmClient,
    private readonly trustLayer: EinsteinTrustLayer,
  ) {}

  async triage(
    finding: Finding,
    ilLevel: IlLevel,
    asOf: Date,
    trace: TraceStep[],
  ): Promise<SubagentResult> {
    // 1. Deterministic actions --------------------------------------------
    const results = runDeterministicActions(finding, asOf);
    for (const r of results) {
      trace.push({
        node: "deterministicActions",
        message: `${r.rule}: ${r.conclusive ? "FIRED" : "no"} — ${r.summary}`,
      });
    }
    const conclusive = results.find((r) => r.conclusive);
    if (conclusive?.conclusive) {
      trace.push({ node: "TriageSubagent", message: "Deterministic result is conclusive. No LLM call." });
      return { kind: "deterministic", result: conclusive };
    }
    trace.push({ node: "TriageSubagent", message: "Deterministic actions inconclusive." });

    // 2. Impact Level gate — IL5 never reaches the model ------------------
    if (!reasoningAllowed(ilLevel)) {
      trace.push({
        node: "TriageSubagent",
        message: `${ilLevel}: LLM reasoning is not authorized at this Impact Level. Skipping reasoning; routing to human.`,
      });
      return { kind: "reasoning_not_permitted", ilLevel };
    }

    // 3. Reasoning action → trust layer ------------------------------------
    trace.push({ node: "reasoningActions", message: `Reviewing submitted justification via ${this.llm.name}.` });
    const { rawOutput, error } = await reviewJustification(finding, this.llm);
    if (error) trace.push({ node: "reasoningActions", message: `LLM call failed: ${error}` });

    const trust = this.trustLayer.evaluate(rawOutput, finding);
    trace.push({
      node: "EinsteinTrustLayer",
      message: trust.passed
        ? `PASSED (schema ok, grounded to ${finding.cveId}). Model says ${trust.verdict.recommendation} @ ${trust.verdict.confidence}.`
        : `FAILED — ${trust.failures.join("; ")}. Forcing escalation.`,
    });
    return { kind: "reasoning", trust, rawOutput };
  }
}
