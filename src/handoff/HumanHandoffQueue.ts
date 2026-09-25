import type { MockCveQueueApi } from "../data/mockCveQueueApi.js";
import { InMemoryStore, type StateStore } from "../data/stateStore.js";
import type {
  AuditLogEntry,
  Finding,
  HandoffQueueItem,
  HandoffReason,
  IlLevel,
  ReviewerAction,
} from "../types.js";

/**
 * Queue of findings waiting on a human reviewer (a CST analyst in this model).
 * Resolving an item writes a `human`-sourced decision to the queue API, so the
 * audit log shows the complete chain: automated routing → human decision.
 */
export class HumanHandoffQueue {
  constructor(
    private readonly queueApi: MockCveQueueApi,
    private readonly store: StateStore<HandoffQueueItem[]> = new InMemoryStore<HandoffQueueItem[]>([]),
    private readonly now: () => Date = () => new Date(),
  ) {}

  enqueue(finding: Finding, reason: HandoffReason, ilLevel: IlLevel, context = ""): HandoffQueueItem {
    const items = this.store.load();
    const item: HandoffQueueItem = {
      findingId: finding.id,
      cveId: finding.cveId,
      severity: finding.severity,
      reason,
      ilLevel,
      enqueuedAt: this.now().toISOString(),
      context,
      status: "pending",
    };
    // One open item per finding: re-triage replaces a stale pending item.
    const kept = items.filter((i) => !(i.findingId === finding.id && i.status === "pending"));
    kept.push(item);
    this.store.save(kept);
    return item;
  }

  list(filter: { status?: HandoffQueueItem["status"] } = {}): HandoffQueueItem[] {
    const items = this.store.load();
    return filter.status ? items.filter((i) => i.status === filter.status) : items;
  }

  /** Simulate a reviewer resolving a pending item. */
  resolve(
    findingId: string,
    action: ReviewerAction,
    note: string,
    reviewer = "cst-reviewer",
  ): { item: HandoffQueueItem; auditEntry: AuditLogEntry } {
    const items = this.store.load();
    const item = items.find((i) => i.findingId === findingId && i.status === "pending");
    if (!item) throw new Error(`No pending handoff item for finding "${findingId}".`);

    item.status = "resolved";
    item.resolution = { action, note, reviewer, resolvedAt: this.now().toISOString() };
    this.store.save(items);

    const auditEntry = this.queueApi.writeDecision(
      findingId,
      {
        outcome: action === "accept" ? "justification_accepted" : "justification_rejected",
        detail: `Reviewed by ${reviewer}: ${note}`,
        handoffReason: item.reason,
      },
      "human",
      item.ilLevel,
      null,
    );
    return { item, auditEntry };
  }

  clear(): void {
    this.store.save([]);
  }
}
