import type { AuditLogEntry, Decision, DecisionSource, IlLevel } from "../types.js";
import { InMemoryStore, type StateStore } from "./stateStore.js";

/**
 * "Data 360 / Apex / APIs" stand-in: the system of record for triage decisions.
 *
 * Every decision — deterministic, model-assisted, or human — is written here as
 * an append-only audit entry that records who decided, at what Impact Level,
 * and whether the trust layer passed. No external calls.
 */
export class MockCveQueueApi {
  constructor(
    private readonly store: StateStore<AuditLogEntry[]> = new InMemoryStore<AuditLogEntry[]>([]),
    private readonly now: () => Date = () => new Date(),
  ) {}

  writeDecision(
    findingId: string,
    decision: Decision,
    source: DecisionSource,
    ilLevel: IlLevel,
    trustLayerPassed: boolean | null,
  ): AuditLogEntry {
    const log = this.store.load();
    const entry: AuditLogEntry = {
      sequence: log.length + 1,
      timestamp: this.now().toISOString(),
      findingId,
      decision,
      source,
      ilLevel,
      trustLayerPassed,
    };
    log.push(entry);
    this.store.save(log);
    return entry;
  }

  getAuditLog(): AuditLogEntry[] {
    return this.store.load();
  }

  clear(): void {
    this.store.save([]);
  }
}
