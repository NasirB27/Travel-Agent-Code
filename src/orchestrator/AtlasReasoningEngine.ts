import type { MockCveQueueApi } from "../data/mockCveQueueApi.js";
import type { MockFindingsStore } from "../data/mockFindingsStore.js";
import type { HumanHandoffQueue } from "../handoff/HumanHandoffQueue.js";
import type { TriageSubagent } from "../subagents/TriageSubagent.js";
import type {
  Decision,
  DecisionOutcome,
  DeterministicDisposition,
  Finding,
  HandoffReason,
  IlLevel,
  SubagentResult,
  TraceStep,
  TriageOutcome,
} from "../types.js";

/**
 * AtlasReasoningEngine: the central router.
 *
 * Loads the finding, routes it to the TriageSubagent, then makes the top-level
 * call: write an automated decision, or hand off to a human. Handoff happens on
 * any of:
 *   - IL5 mode (reasoning not permitted)
 *   - trust layer failure
 *   - severity === "Critical" (always gets human eyes, even with a clean automated answer)
 *   - the model itself recommending escalation
 *   - model confidence below threshold
 */

export const DEFAULT_CONFIDENCE_THRESHOLD = 0.75;

export interface AtlasReasoningEngineDeps {
  findingsStore: MockFindingsStore;
  subagent: TriageSubagent;
  queueApi: MockCveQueueApi;
  handoffQueue: HumanHandoffQueue;
  ilLevel: IlLevel;
  confidenceThreshold?: number;
}

const DISPOSITION_OUTCOME: Record<DeterministicDisposition, DecisionOutcome> = {
  accepted_risk_low_severity: "auto_closed_accepted_risk",
  patch_available_recommend_rebuild: "flagged_patch_available",
  stale_base_image: "flagged_stale_base_image",
};

/**
 * Pure routing rule: should this result go to a human, and why?
 * Checks run in priority order; the first match is the recorded reason.
 */
export function determineHandoffReason(
  finding: Finding,
  result: SubagentResult,
  confidenceThreshold: number,
): HandoffReason | null {
  if (result.kind === "reasoning_not_permitted") return "il5_mode";
  if (result.kind === "reasoning" && !result.trust.passed) return "trust_layer_failed";
  if (finding.severity === "Critical") return "critical_severity";
  if (result.kind === "reasoning") {
    if (result.trust.verdict.recommendation === "escalate") return "model_recommended_escalation";
    if (result.trust.verdict.confidence < confidenceThreshold) return "low_confidence";
  }
  return null;
}

/** What the automated path concluded, independent of whether a human must confirm it. */
function describeAutomatedResult(result: SubagentResult): string {
  switch (result.kind) {
    case "deterministic":
      return result.result.summary;
    case "reasoning": {
      const v = result.trust.verdict;
      return `Model: ${v.recommendation} (confidence ${v.confidence}) — ${v.rationale}`;
    }
    case "reasoning_not_permitted":
      return `Deterministic rules inconclusive; LLM reasoning not authorized at ${result.ilLevel}.`;
  }
}

export class AtlasReasoningEngine {
  private readonly confidenceThreshold: number;

  constructor(private readonly deps: AtlasReasoningEngineDeps) {
    this.confidenceThreshold = deps.confidenceThreshold ?? DEFAULT_CONFIDENCE_THRESHOLD;
  }

  get ilLevel(): IlLevel {
    return this.deps.ilLevel;
  }

  async triage(findingId: string): Promise<TriageOutcome> {
    const { findingsStore, subagent, queueApi, handoffQueue, ilLevel } = this.deps;
    const trace: TraceStep[] = [];

    const finding = findingsStore.getById(findingId);
    trace.push({
      node: "AtlasReasoningEngine",
      message: `Loaded ${finding.id} (${finding.cveId}, ${finding.severity}, ${finding.packageName}) at ${ilLevel}. Routing to TriageSubagent.`,
    });

    const result = await subagent.triage(finding, ilLevel, findingsStore.snapshotDate, trace);
    const trustLayerPassed = result.kind === "reasoning" ? result.trust.passed : null;
    const handoffReason = determineHandoffReason(finding, result, this.confidenceThreshold);

    // Parallel path: human handoff -----------------------------------------
    if (handoffReason) {
      const context = describeAutomatedResult(result);
      handoffQueue.enqueue(finding, handoffReason, ilLevel, context);
      trace.push({ node: "AtlasReasoningEngine", message: `Handoff required: ${handoffReason}.` });
      trace.push({ node: "HumanHandoffQueue", message: `Enqueued ${finding.id} for CST review.` });

      const decision: Decision = {
        outcome: "pending_human_review",
        detail: context,
        handoffReason,
        ...(result.kind === "reasoning" && {
          recommendation: result.trust.verdict.recommendation,
          confidence: result.trust.verdict.confidence,
        }),
      };
      // Attribute the routing to the stage that actually ran: if no model output
      // exists (deterministic hit, or IL5 skip), the deterministic stage routed it.
      const source = result.kind === "reasoning" ? "reasoning" : "deterministic";
      queueApi.writeDecision(finding.id, decision, source, ilLevel, trustLayerPassed);
      trace.push({ node: "mockCveQueueApi", message: "Audit entry written: pending_human_review." });
      return { finding, ilLevel, subagentResult: result, decision, source, handoffReason, trace };
    }

    // Automated decision ---------------------------------------------------
    let decision: Decision;
    if (result.kind === "deterministic") {
      decision = { outcome: DISPOSITION_OUTCOME[result.result.disposition], detail: result.result.summary };
    } else if (result.kind === "reasoning") {
      const v = result.trust.verdict;
      decision = {
        outcome: v.recommendation === "accept" ? "justification_accepted" : "justification_rejected",
        detail: v.rationale,
        recommendation: v.recommendation,
        confidence: v.confidence,
      };
    } else {
      // Unreachable: determineHandoffReason always hands off reasoning_not_permitted.
      throw new Error("Invariant violated: reasoning_not_permitted must route to handoff.");
    }

    const source = result.kind;
    queueApi.writeDecision(finding.id, decision, source, ilLevel, trustLayerPassed);
    trace.push({ node: "mockCveQueueApi", message: `Audit entry written: ${decision.outcome} (source: ${source}).` });
    return { finding, ilLevel, subagentResult: result, decision, source, handoffReason: null, trace };
  }

  async triageAll(): Promise<TriageOutcome[]> {
    const outcomes: TriageOutcome[] = [];
    for (const finding of this.deps.findingsStore.list()) {
      outcomes.push(await this.triage(finding.id));
    }
    return outcomes;
  }
}
