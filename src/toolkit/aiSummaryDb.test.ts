import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { readLocalAiSummary } from "./aiSummaryDb";

describe("readLocalAiSummary", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("本地数据库不存在时读取静态摘要构建产物", () => {
    vi.stubEnv("AI_SUMMARY_DB_PATH", join(process.cwd(), ".hyacine", "missing-ai-summary-test.db"));

    const summary = readLocalAiSummary("CS61B/CS61B-CHAPTER-1.mdx", "CS61B CHAPTER 1");

    expect(summary?.content).toContain("Java");
    expect(summary?.model).toBe("ecnu-max");
  });

  it("找不到文章摘要时返回 null", () => {
    vi.stubEnv("AI_SUMMARY_DB_PATH", join(process.cwd(), ".hyacine", "missing-ai-summary-test.db"));

    expect(readLocalAiSummary("not-found-post")).toBeNull();
  });
});
