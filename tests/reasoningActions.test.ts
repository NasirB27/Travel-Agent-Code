import { describe, expect, it } from "vitest";
import {
  buildJustificationReviewPrompt,
  reviewJustification,
  VERDICT_JSON_SCHEMA,
} from "../src/actions/reasoningActions.js";
import { makeFinding, mockLlm } from "./helpers.js";

describe("buildJustificationReviewPrompt", () => {
  it("includes every field the model needs to judge the justification", () => {
    const f = makeFinding({ justification: "Custom justification text." });
    const prompt = buildJustificationReviewPrompt(f);
    for (const expected of [
      f.cveId,
      f.packageName,
      f.severity,
      String(f.cvss),
      String(f.epss),
      f.imageName,
      "Custom justification text.",
    ]) {
      expect(prompt).toContain(expected);
    }
  });
});

describe("reviewJustification", () => {
  it("calls the LLM once with the prompt and the verdict schema, and returns raw text untouched", async () => {
    const raw = '{"recommendation":"accept","confidence":0.9,"rationale":"ok","referencedCveId":"X"}';
    const { client, complete } = mockLlm(() => raw);
    const finding = makeFinding();

    const result = await reviewJustification(finding, client);

    expect(complete).toHaveBeenCalledTimes(1);
    const req = complete.mock.calls[0]![0] as { prompt: string; jsonSchema: unknown };
    expect(req.prompt).toContain(finding.cveId);
    expect(req.jsonSchema).toBe(VERDICT_JSON_SCHEMA);
    // No parsing or trust decisions here — that is the trust layer's job.
    expect(result).toEqual({ rawOutput: raw });
  });

  it("turns a transport failure into empty output (so the trust layer fails it closed)", async () => {
    const { client } = mockLlm(() => {
      throw new Error("network down");
    });
    const result = await reviewJustification(makeFinding(), client);
    expect(result).toEqual({ rawOutput: "", error: "network down" });
  });
});
