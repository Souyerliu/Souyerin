import { join } from "node:path";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { DatabaseSync } from "node:sqlite";

import { afterEach, describe, expect, it, vi } from "vitest";

import { readLocalAiSummary } from "./aiSummaryDb";

describe("readLocalAiSummary", () => {
  let fixtureDirectory: string | undefined;
  afterEach(() => {
    vi.unstubAllEnvs();
    if (fixtureDirectory) {
      rmSync(fixtureDirectory, { recursive: true, force: true });
      fixtureDirectory = undefined;
    }
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

  it("静态摘要优先匹配当前路径，避免同名旧文章覆盖", () => {
    vi.stubEnv("AI_SUMMARY_DB_PATH", join(process.cwd(), ".hyacine", "missing-ai-summary-test.db"));

    const summary = readLocalAiSummary("computer-science/cs61b/cs61b-chapter-7", "CS61B CHAPTER 7");

    expect(summary?.content).toContain("Java面向对象");
    expect(summary?.content).not.toContain("人工智能技术");
  });

  it("数据库优先匹配路径，同时保留标题回退", () => {
    fixtureDirectory = mkdtempSync(join(tmpdir(), "ai-summary-"));
    const path = join(fixtureDirectory, "data.db");
    const database = new DatabaseSync(path);
    try {
      database.exec(
        "CREATE TABLE Post (path TEXT, title TEXT, summary TEXT, summaryModel TEXT, summarySourceHash TEXT)",
      );
      const insert = database.prepare("INSERT INTO Post VALUES (?, ?, ?, ?, ?)");
      insert.run("@/src/posts/old.mdx", "同名文章", "旧路径摘要", "test", null);
      insert.run("@/src/posts/new.mdx", "同名文章", "当前路径摘要", "test", null);
    } finally {
      database.close();
    }
    vi.stubEnv("AI_SUMMARY_DB_PATH", path);

    expect(readLocalAiSummary("NEW", "同名文章")?.content).toBe("当前路径摘要");
    expect(readLocalAiSummary("new")?.content).toBe("当前路径摘要");
    expect(readLocalAiSummary("missing", "同名文章")?.content).toBe("旧路径摘要");
  });
});
