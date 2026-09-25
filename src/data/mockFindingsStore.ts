import { readFileSync } from "node:fs";
import type { Finding } from "../types.js";

/**
 * "Data 360" stand-in: read-only access to scan findings.
 *
 * The dataset carries a `snapshotDate` — the date the scan was taken — which is
 * the reference point for time-based rules like base-image freshness. That keeps
 * the demo output identical no matter when it is run.
 */
export interface FindingsDataset {
  snapshotDate: string;
  findings: Finding[];
}

export class MockFindingsStore {
  private readonly byId: Map<string, Finding>;
  readonly snapshotDate: Date;

  constructor(dataset: FindingsDataset) {
    this.snapshotDate = new Date(dataset.snapshotDate);
    if (Number.isNaN(this.snapshotDate.getTime())) {
      throw new Error(`Invalid snapshotDate "${dataset.snapshotDate}".`);
    }
    this.byId = new Map(dataset.findings.map((f) => [f.id, f]));
  }

  static fromFile(path: string): MockFindingsStore {
    return new MockFindingsStore(JSON.parse(readFileSync(path, "utf8")) as FindingsDataset);
  }

  getById(id: string): Finding {
    const finding = this.byId.get(id);
    if (!finding) throw new Error(`Finding "${id}" not found.`);
    return finding;
  }

  list(): Finding[] {
    return [...this.byId.values()];
  }
}
