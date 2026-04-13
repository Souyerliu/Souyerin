import { describe, expect, it } from "bun:test";
import { resolveLocale } from "./resolveLocale";

describe("resolveLocale", () => {
  it("returns configured locale when valid", () => {
    expect(resolveLocale("zh-CN")).toBe("zh-CN");
    expect(resolveLocale("en")).toBe("en");
  });

  it("falls back to zh-CN for missing or invalid locale", () => {
    expect(resolveLocale(undefined)).toBe("zh-CN");
    expect(resolveLocale(null)).toBe("zh-CN");
    expect(resolveLocale("ja")).toBe("zh-CN");
  });

  it("supports custom fallback locale", () => {
    expect(resolveLocale("ja", "en")).toBe("en");
  });
});
