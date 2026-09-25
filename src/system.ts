import { MockCveQueueApi } from "./data/mockCveQueueApi.js";
import { MockFindingsStore } from "./data/mockFindingsStore.js";
import { InMemoryStore, type StateStore } from "./data/stateStore.js";
import { HumanHandoffQueue } from "./handoff/HumanHandoffQueue.js";
import type { LlmClient } from "./llm/LlmClient.js";
import { AtlasReasoningEngine } from "./orchestrator/AtlasReasoningEngine.js";
import { TriageSubagent } from "./subagents/TriageSubagent.js";
import { EinsteinTrustLayer } from "./trust/EinsteinTrustLayer.js";
import type { AuditLogEntry, HandoffQueueItem, IlLevel } from "./types.js";

/** Composition root: wires every node in the diagram together. */
export interface TriageSystemOptions {
  findingsStore: MockFindingsStore;
  llm: LlmClient;
  ilLevel: IlLevel;
  auditStore?: StateStore<AuditLogEntry[]>;
  handoffStore?: StateStore<HandoffQueueItem[]>;
  confidenceThreshold?: number;
}

export interface TriageSystem {
  engine: AtlasReasoningEngine;
  queueApi: MockCveQueueApi;
  handoffQueue: HumanHandoffQueue;
  findingsStore: MockFindingsStore;
}

export function createTriageSystem(options: TriageSystemOptions): TriageSystem {
  const queueApi = new MockCveQueueApi(options.auditStore ?? new InMemoryStore<AuditLogEntry[]>([]));
  const handoffQueue = new HumanHandoffQueue(
    queueApi,
    options.handoffStore ?? new InMemoryStore<HandoffQueueItem[]>([]),
  );
  const subagent = new TriageSubagent(options.llm, new EinsteinTrustLayer());
  const engine = new AtlasReasoningEngine({
    findingsStore: options.findingsStore,
    subagent,
    queueApi,
    handoffQueue,
    ilLevel: options.ilLevel,
    confidenceThreshold: options.confidenceThreshold,
  });
  return { engine, queueApi, handoffQueue, findingsStore: options.findingsStore };
}
