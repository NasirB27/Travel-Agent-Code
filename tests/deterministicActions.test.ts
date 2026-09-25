import { describe, expect, it } from "vitest";
import {
  checkBaseImageFreshness,
  checkKnownFixAvailable,
  checkSeverityThreshold,
  runDeterministicActions,
} from "../src/actions/deterministicActions.js";
import { AS_OF, makeFinding } from "./helpers.js";

describe("checkSeverityThreshold", () => {
  it("auto-closes when CVSS < 4.0 AND EPSS < 0.1", () => {
    const r = checkSeverityThreshold(makeFinding({ cvss: 3.9, epss: 0.09 }));
    expect(r).toMatchObject({ conclusive: true, disposition: "accepted_risk_low_severity" });
  });

  it("is inconclusive when only CVSS is low", () => {
    expect(checkSeverityThreshold(makeFinding({ cvss: 2.0, epss: 0.5 })).conclusive).toBe(false);
  });

  it("is inconclusive when only EPSS is low", () => {
    expect(checkSeverityThreshold(makeFinding({ cvss: 8.0, epss: 0.01 })).conclusive).toBe(false);
  });

  it("treats the thresholds as strict (4.0 / 0.1 do not qualify)", () => {
    expect(checkSeverityThreshold(makeFinding({ cvss: 4.0, epss: 0.01 })).conclusive).toBe(false);
    expect(checkSeverityThreshold(makeFinding({ cvss: 1.0, epss: 0.1 })).conclusive).toBe(false);
  });
});

describe("checkKnownFixAvailable", () => {
  it("flags a rebuild when a fixed version exists and the image is not on it", () => {
    const r = checkKnownFixAvailable(makeFinding({ installedVersion: "1.0.0", fixedVersion: "1.0.1" }));
    expect(r).toMatchObject({ conclusive: true, disposition: "patch_available_recommend_rebuild" });
  });

  it("is inconclusive when no fix exists", () => {
    expect(checkKnownFixAvailable(makeFinding({ fixedVersion: null })).conclusive).toBe(false);
  });

  it("is inconclusive when already pinned to the fixed version", () => {
    const r = checkKnownFixAvailable(makeFinding({ installedVersion: "1.0.1", fixedVersion: "1.0.1" }));
    expect(r.conclusive).toBe(false);
  });
});

describe("checkBaseImageFreshness", () => {
  it("flags a base image rebuilt more than 90 days ago", () => {
    const r = checkBaseImageFreshness(makeFinding({ baseImageLastRebuilt: "2026-06-01" }), AS_OF);
    expect(r).toMatchObject({ conclusive: true, disposition: "stale_base_image" });
  });

  it("does not flag at exactly 90 days", () => {
    // 2026-06-17 → 2026-09-15 is exactly 90 days.
    const r = checkBaseImageFreshness(makeFinding({ baseImageLastRebuilt: "2026-06-17" }), AS_OF);
    expect(r.conclusive).toBe(false);
  });

  it("is inconclusive for a fresh base image", () => {
    expect(checkBaseImageFreshness(makeFinding({ baseImageLastRebuilt: "2026-09-01" }), AS_OF).conclusive).toBe(false);
  });

  it("rejects an unparseable date instead of guessing", () => {
    expect(() => checkBaseImageFreshness(makeFinding({ baseImageLastRebuilt: "not-a-date" }), AS_OF)).toThrow();
  });
});

describe("runDeterministicActions", () => {
  it("stops at the first conclusive rule", () => {
    const results = runDeterministicActions(
      makeFinding({ cvss: 1.0, epss: 0.01, fixedVersion: "9.9.9", baseImageLastRebuilt: "2020-01-01" }),
      AS_OF,
    );
    expect(results).toHaveLength(1);
    expect(results[0]).toMatchObject({ conclusive: true, rule: "severity_threshold" });
  });

  it("falls through with every rule inconclusive", () => {
    const results = runDeterministicActions(makeFinding(), AS_OF);
    expect(results.map((r) => r.rule)).toEqual(["severity_threshold", "known_fix_available", "base_image_freshness"]);
    expect(results.every((r) => !r.conclusive)).toBe(true);
  });
});
