import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { readIlLevelFromEnv, reasoningAllowed } from "./config/ilLevel.js";
import { MockFindingsStore } from "./data/mockFindingsStore.js";
import { JsonFileStore } from "./data/stateStore.js";
import { createLlmClientFromEnv, type LlmClient } from "./llm/LlmClient.js";
import { createTriageSystem, type TriageSystem } from "./system.js";
import type {
  AuditLogEntry,
  HandoffQueueItem,
  HandoffReason,
  IlLevel,
  ReviewerAction,
  TriageOutcome,
} from "./types.js";

const ROOT = resolve(import.meta.dirname, "..");
const FINDINGS_PATH = resolve(ROOT, "mock-data/findings.json");
const REPLAY_PATH = resolve(ROOT, "mock-data/recorded-llm-responses.json");
const STATE_DIR = resolve(ROOT, ".state");

// Load .env if present. Real environment variables take precedence, so
// `IL_LEVEL=IL5 npm run cli -- ...` always wins over the file.
const ENV_PATH = resolve(ROOT, ".env");
if (existsSync(ENV_PATH)) process.loadEnvFile(ENV_PATH);

const USAGE = `
ironbank-atlas-triage — illustrative CVE triage workflow (synthetic data)

Usage: npm run cli -- <command> [args]

Commands:
  triage <findingId>                    Run one finding through the full flow and print the decision path
  triage-all                            Run every finding (fresh batch) and print a summary table
  queue [--all]                         Print the human handoff queue (pending only unless --all)
  resolve <findingId> <accept|reject>   Simulate a CST reviewer resolving a queued item
          [--note "..."] [--reviewer name]
  audit-log                             Print the full decision audit log
  reset                                 Clear the local audit log and handoff queue

Environment:
  IL_LEVEL=IL2|IL4|IL5   Impact Level (default IL4). IL5 disables LLM reasoning.
  ANTHROPIC_API_KEY      Live Claude calls. Unset → offline replay of recorded responses.
  LLM_MODE=offline       Force offline replay even when a key is set.

Example:
  IL_LEVEL=IL5 npm run cli -- triage-all
`.trim();

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

const useColor = process.stdout.isTTY && !process.env.NO_COLOR;
const paint = (code: number) => (s: string) => (useColor ? `\x1b[${code}m${s}\x1b[0m` : s);
const bold = paint(1);
const dim = paint(2);
const red = paint(31);
const green = paint(32);
const yellow = paint(33);
const cyan = paint(36);

function table(headers: string[], rows: string[][]): string {
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => (r[i] ?? "").length)));
  const line = (cells: string[]) =>
    cells.map((c, i) => c.padEnd(widths[i] ?? 0)).join("  ").trimEnd();
  return [bold(line(headers)), widths.map((w) => "─".repeat(w)).join("  "), ...rows.map(line)].join("\n");
}

function banner(ilLevel: IlLevel, llm: LlmClient): void {
  const reasoning = reasoningAllowed(ilLevel)
    ? green(`enabled (${llm.name})`)
    : red("DISABLED — not authorized at this Impact Level");
  console.log(`${bold("Impact Level:")} ${ilLevel}   ${bold("LLM reasoning:")} ${reasoning}`);
  console.log(dim("Synthetic, illustrative data only. Not affiliated with Salesforce or DoD.\n"));
}

function outcomeLabel(o: TriageOutcome): string {
  if (o.handoffReason) return yellow(`HUMAN HANDOFF (${o.handoffReason})`);
  return green(`${o.decision.outcome} [${o.source}]`);
}

function printTrace(o: TriageOutcome): void {
  const f = o.finding;
  console.log(bold(`${f.id}  ${f.cveId}  ${f.severity}  ${f.packageName} ${f.installedVersion}  ${f.imageName}:${f.imageTag}`));
  console.log(dim("Decision path:"));
  o.trace.forEach((step, i) => {
    console.log(`  ${String(i + 1).padStart(2)}. ${cyan(step.node.padEnd(21))} ${step.message}`);
  });
  console.log(`${bold("Result:")} ${outcomeLabel(o)}\n`);
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

async function cmdTriage(system: TriageSystem, findingId: string | undefined): Promise<void> {
  if (!findingId) throw new Error("Usage: triage <findingId>");
  printTrace(await system.engine.triage(findingId));
}

async function cmdTriageAll(system: TriageSystem): Promise<void> {
  system.queueApi.clear();
  system.handoffQueue.clear();
  const outcomes = await system.engine.triageAll();

  console.log(
    table(
      ["Finding", "CVE", "Severity", "Path", "Result"],
      outcomes.map((o) => [
        o.finding.id,
        o.finding.cveId,
        o.finding.severity,
        o.subagentResult.kind,
        o.handoffReason ? `HANDOFF: ${o.handoffReason}` : `${o.decision.outcome}`,
      ]),
    ),
  );

  const autoDeterministic = outcomes.filter((o) => !o.handoffReason && o.source === "deterministic").length;
  const autoReasoning = outcomes.filter((o) => !o.handoffReason && o.source === "reasoning").length;
  const forcedEscalations = outcomes.filter(
    (o) => o.subagentResult.kind === "reasoning" && !o.subagentResult.trust.passed,
  ).length;
  const llmCalls = outcomes.filter((o) => o.subagentResult.kind === "reasoning").length;
  const byReason = new Map<HandoffReason, number>();
  for (const o of outcomes) {
    if (o.handoffReason) byReason.set(o.handoffReason, (byReason.get(o.handoffReason) ?? 0) + 1);
  }
  const handoffTotal = [...byReason.values()].reduce((a, b) => a + b, 0);

  console.log(`\n${bold("Summary")} (${outcomes.length} findings, ${llmCalls} LLM calls)`);
  const rows: string[][] = [
    ["Auto-resolved — deterministic", String(autoDeterministic)],
    ["Auto-resolved — reasoning (trust layer passed)", String(autoReasoning)],
    ["Escalated — forced by trust layer", String(forcedEscalations)],
    ["Human handoff — total", String(handoffTotal)],
    ...[...byReason.entries()].map(([reason, n]) => [`  └ ${reason}`, String(n)]),
  ];
  console.log(table(["Bucket", "Count"], rows));
  console.log(dim(`\nState written to ${STATE_DIR}. Try: npm run cli -- queue`));
}

function cmdQueue(system: TriageSystem, all: boolean): void {
  const items = system.handoffQueue.list(all ? {} : { status: "pending" });
  if (items.length === 0) {
    console.log("Handoff queue is empty.");
    return;
  }
  console.log(
    table(
      ["Finding", "CVE", "Severity", "Reason", "Status", "Context"],
      items.map((i: HandoffQueueItem) => [
        i.findingId,
        i.cveId,
        i.severity,
        i.reason,
        i.status === "resolved" && i.resolution ? `${i.resolution.action}ed` : i.status,
        truncate(i.context, 70),
      ]),
    ),
  );
}

function cmdResolve(
  system: TriageSystem,
  findingId: string | undefined,
  action: string | undefined,
  note: string,
  reviewer: string,
): void {
  if (!findingId || (action !== "accept" && action !== "reject")) {
    throw new Error('Usage: resolve <findingId> <accept|reject> [--note "..."] [--reviewer name]');
  }
  const { auditEntry } = system.handoffQueue.resolve(findingId, action as ReviewerAction, note, reviewer);
  console.log(green(`Resolved ${findingId}: ${auditEntry.decision.outcome} by ${reviewer}. Audit entry #${auditEntry.sequence} written.`));
}

function cmdAuditLog(system: TriageSystem): void {
  const log = system.queueApi.getAuditLog();
  if (log.length === 0) {
    console.log("Audit log is empty.");
    return;
  }
  console.log(
    table(
      ["#", "Finding", "Source", "IL", "Trust", "Outcome", "Detail"],
      log.map((e: AuditLogEntry) => [
        String(e.sequence),
        e.findingId,
        e.source,
        e.ilLevel,
        e.trustLayerPassed === null ? "n/a" : e.trustLayerPassed ? "pass" : "FAIL",
        e.decision.handoffReason ? `${e.decision.outcome} (${e.decision.handoffReason})` : e.decision.outcome,
        truncate(e.decision.detail, 60),
      ]),
    ),
  );
}

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const { positionals, values } = parseArgs({
    allowPositionals: true,
    options: {
      note: { type: "string", default: "Reviewed by CST." },
      reviewer: { type: "string", default: "cst-reviewer" },
      all: { type: "boolean", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
  });
  const [command, ...args] = positionals;
  if (!command || values.help || command === "help") {
    console.log(USAGE);
    return;
  }

  const ilLevel = readIlLevelFromEnv();
  const llm = createLlmClientFromEnv(REPLAY_PATH);
  const system = createTriageSystem({
    findingsStore: MockFindingsStore.fromFile(FINDINGS_PATH),
    llm,
    ilLevel,
    auditStore: new JsonFileStore<AuditLogEntry[]>(resolve(STATE_DIR, "audit-log.json"), []),
    handoffStore: new JsonFileStore<HandoffQueueItem[]>(resolve(STATE_DIR, "handoff-queue.json"), []),
  });

  switch (command) {
    case "triage":
      banner(ilLevel, llm);
      return cmdTriage(system, args[0]);
    case "triage-all":
      banner(ilLevel, llm);
      return cmdTriageAll(system);
    case "queue":
      return cmdQueue(system, values.all);
    case "resolve":
      return cmdResolve(system, args[0], args[1], values.note, values.reviewer);
    case "audit-log":
      return cmdAuditLog(system);
    case "reset":
      system.queueApi.clear();
      system.handoffQueue.clear();
      console.log("Cleared audit log and handoff queue.");
      return;
    default:
      throw new Error(`Unknown command "${command}".\n\n${USAGE}`);
  }
}

main().catch((err: unknown) => {
  console.error(red(err instanceof Error ? err.message : String(err)));
  process.exitCode = 1;
});
