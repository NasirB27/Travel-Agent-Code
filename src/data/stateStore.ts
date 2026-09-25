import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

/**
 * Minimal persistence seam. Tests use the in-memory store; the CLI uses a JSON
 * file so `queue` and `audit-log` can be inspected across separate invocations.
 */
export interface StateStore<T> {
  load(): T;
  save(value: T): void;
}

export class InMemoryStore<T> implements StateStore<T> {
  private value: T;
  constructor(initial: T) {
    this.value = structuredClone(initial);
  }
  load(): T {
    return structuredClone(this.value);
  }
  save(value: T): void {
    this.value = structuredClone(value);
  }
}

export class JsonFileStore<T> implements StateStore<T> {
  constructor(
    private readonly path: string,
    private readonly initial: T,
  ) {}
  load(): T {
    if (!existsSync(this.path)) return structuredClone(this.initial);
    return JSON.parse(readFileSync(this.path, "utf8")) as T;
  }
  save(value: T): void {
    mkdirSync(dirname(this.path), { recursive: true });
    writeFileSync(this.path, JSON.stringify(value, null, 2) + "\n");
  }
}
