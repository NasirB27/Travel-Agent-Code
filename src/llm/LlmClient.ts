import { readFileSync } from "node:fs";
import Anthropic from "@anthropic-ai/sdk";

/**
 * Thin, provider-agnostic LLM interface.
 *
 * The rest of the system only ever sees `LlmClient`. It hands over a prompt and
 * a JSON schema and gets back raw text. Parsing and validation are deliberately
 * NOT done here — that is the Einstein Trust Layer's job, and it must run no
 * matter which provider produced the text.
 */
export interface LlmRequest {
  system: string;
  prompt: string;
  /** JSON schema the response should conform to. Providers may enforce it natively. */
  jsonSchema: Record<string, unknown>;
}

export interface LlmClient {
  /** Human-readable provider label for traces, e.g. "anthropic:claude-opus-5". */
  readonly name: string;
  /** Returns the model's raw text output. May throw on transport errors. */
  complete(request: LlmRequest): Promise<string>;
}

// ---------------------------------------------------------------------------
// Anthropic (Claude)
// ---------------------------------------------------------------------------

export const DEFAULT_ANTHROPIC_MODEL = "claude-opus-5";

export interface AnthropicLlmClientOptions {
  /** Defaults to the SDK's own resolution (ANTHROPIC_API_KEY, etc.). Never hardcode. */
  apiKey?: string;
  model?: string;
}

export class AnthropicLlmClient implements LlmClient {
  readonly name: string;
  private readonly client: Anthropic;
  private readonly model: string;

  constructor(options: AnthropicLlmClientOptions = {}) {
    this.model = options.model || DEFAULT_ANTHROPIC_MODEL;
    this.client = new Anthropic(options.apiKey ? { apiKey: options.apiKey } : {});
    this.name = `anthropic:${this.model}`;
  }

  async complete(request: LlmRequest): Promise<string> {
    const response = await this.client.beta.messages.create({
      model: this.model,
      max_tokens: 16000,
      system: request.system,
      messages: [{ role: "user", content: request.prompt }],
      output_config: {
        effort: "medium",
        format: { type: "json_schema", schema: request.jsonSchema },
      },
      // Server-side refusal fallback: if the primary model declines, the API
      // re-runs the request on a fallback model inside the same call.
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
    });

    // A refusal yields no usable verdict. Return empty text; the trust layer's
    // schema check will fail it closed and force escalation to a human.
    if (response.stop_reason === "refusal") return "";

    return response.content
      .flatMap((block) => (block.type === "text" ? [block.text] : []))
      .join("");
  }
}

// ---------------------------------------------------------------------------
// Offline replay (no network, no key)
// ---------------------------------------------------------------------------

/**
 * Serves pre-recorded model responses keyed by CVE ID, so the whole flow can be
 * demoed with no API key and no network. Responses are stored as raw strings
 * on purpose: some fixtures are deliberately hallucinated or malformed to
 * exercise the trust layer.
 */
export class ReplayLlmClient implements LlmClient {
  readonly name = "replay:recorded-responses";

  constructor(private readonly responsesByCveId: Record<string, string>) {}

  static fromFile(path: string): ReplayLlmClient {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
    const responses: Record<string, string> = {};
    for (const [cveId, value] of Object.entries(parsed)) {
      if (cveId.startsWith("_")) continue; // allow "_comment" keys
      responses[cveId] = typeof value === "string" ? value : JSON.stringify(value);
    }
    return new ReplayLlmClient(responses);
  }

  async complete(request: LlmRequest): Promise<string> {
    const cveId = request.prompt.match(/CVE-\d{4}-\d{4,}/)?.[0];
    const recorded = cveId ? this.responsesByCveId[cveId] : undefined;
    if (recorded !== undefined) return recorded;
    // No recording: behave like a cautious model that declines to decide.
    return JSON.stringify({
      recommendation: "escalate",
      confidence: 0.5,
      rationale: "No recorded response for this finding; deferring to a human reviewer.",
      referencedCveId: cveId ?? "UNKNOWN",
    });
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createLlmClientFromEnv(
  replayFixturePath: string,
  env: NodeJS.ProcessEnv = process.env,
): LlmClient {
  const forceOffline = env.LLM_MODE?.trim().toLowerCase() === "offline";
  if (!forceOffline && env.ANTHROPIC_API_KEY) {
    return new AnthropicLlmClient({ model: env.ANTHROPIC_MODEL?.trim() || undefined });
  }
  return ReplayLlmClient.fromFile(replayFixturePath);
}
