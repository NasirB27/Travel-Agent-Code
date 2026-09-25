/**
 * Shared domain types for the triage workflow.
 *
 * Everything in this repo is a fictional, illustrative domain model. Findings,
 * CVE IDs, and images are synthetic.
 */

export type IlLevel = "IL2" | "IL4" | "IL5";

export type Severity = "Low" | "Medium" | "High" | "Critical";

/** A single vulnerability finding reported against a hardened container image. */
export interface Finding {
  id: string;
  cveId: string;
  packageName: string;
  installedVersion: string;
  /** Version that remediates the CVE, or null if no upstream fix exists yet. */
  fixedVersion: string | null;
  severity: Severity;
  /** CVSS v3 base score, 0.0–10.0. */
  cvss: number;
  /** EPSS exploit-probability score, 0.0–1.0. */
  epss: number;
  imageName: string;
  imageTag: string;
  /** ISO-8601 date the image's base layer was last rebuilt. */
  baseImageLastRebuilt: string;
  /** Free-text justification submitted by the image maintainer. */
  justification: string;
}

// ---------------------------------------------------------------------------
// Deterministic actions
// ---------------------------------------------------------------------------

export type DeterministicRule =
  | "severity_threshold"
  | "known_fix_available"
  | "base_image_freshness";

export type DeterministicDisposition =
  | "accepted_risk_low_severity"
  | "patch_available_recommend_rebuild"
  | "stale_base_image";

export type DeterministicResult =
  | {
      conclusive: true;
      rule: DeterministicRule;
      disposition: DeterministicDisposition;
      summary: string;
    }
  | {
      conclusive: false;
      rule: DeterministicRule;
      summary: string;
    };

// ---------------------------------------------------------------------------
// Reasoning actions + trust layer
// ---------------------------------------------------------------------------

export type Recommendation = "accept" | "reject" | "escalate";

/** The exact shape the reasoning action asks the LLM to return. */
export interface ReasoningVerdict {
  recommendation: Recommendation;
  confidence: number;
  rationale: string;
  referencedCveId: string;
}

/** Result of the Einstein Trust Layer gate on one reasoning output. */
export interface TrustLayerResult {
  passed: boolean;
  groundingPassed: boolean;
  schemaPassed: boolean;
  failures: string[];
  /**
   * The verdict allowed to flow downstream. On failure this is always a forced
   * escalation, regardless of what the model originally said.
   */
  verdict: ReasoningVerdict;
}

// ---------------------------------------------------------------------------
// Subagent → orchestrator contract
// ---------------------------------------------------------------------------

export type SubagentResult =
  | { kind: "deterministic"; result: Extract<DeterministicResult, { conclusive: true }> }
  | { kind: "reasoning"; trust: TrustLayerResult; rawOutput: string }
  | { kind: "reasoning_not_permitted"; ilLevel: IlLevel };

// ---------------------------------------------------------------------------
// Decisions, handoff, audit
// ---------------------------------------------------------------------------

export type HandoffReason =
  | "trust_layer_failed"
  | "il5_mode"
  | "low_confidence"
  | "critical_severity"
  | "model_recommended_escalation";

export type DecisionSource = "deterministic" | "reasoning" | "human";

export type DecisionOutcome =
  | "auto_closed_accepted_risk"
  | "flagged_patch_available"
  | "flagged_stale_base_image"
  | "justification_accepted"
  | "justification_rejected"
  | "pending_human_review";

export interface Decision {
  outcome: DecisionOutcome;
  detail: string;
  recommendation?: Recommendation;
  confidence?: number;
  handoffReason?: HandoffReason;
}

export interface AuditLogEntry {
  sequence: number;
  timestamp: string;
  findingId: string;
  decision: Decision;
  source: DecisionSource;
  ilLevel: IlLevel;
  /** null when no reasoning output passed through the trust layer. */
  trustLayerPassed: boolean | null;
}

export type ReviewerAction = "accept" | "reject";

export interface HandoffQueueItem {
  findingId: string;
  cveId: string;
  severity: Severity;
  reason: HandoffReason;
  /** Impact Level the finding was triaged at; the human decision is recorded under it. */
  ilLevel: IlLevel;
  enqueuedAt: string;
  /** What the automated path would have done, for reviewer context. */
  context: string;
  status: "pending" | "resolved";
  resolution?: {
    action: ReviewerAction;
    note: string;
    reviewer: string;
    resolvedAt: string;
  };
}

/** One node firing in the decision path, for the CLI trace. */
export interface TraceStep {
  node:
    | "AtlasReasoningEngine"
    | "TriageSubagent"
    | "deterministicActions"
    | "reasoningActions"
    | "EinsteinTrustLayer"
    | "mockCveQueueApi"
    | "HumanHandoffQueue";
  message: string;
}

export interface TriageOutcome {
  finding: Finding;
  ilLevel: IlLevel;
  subagentResult: SubagentResult;
  decision: Decision;
  source: DecisionSource;
  handoffReason: HandoffReason | null;
  trace: TraceStep[];
}
