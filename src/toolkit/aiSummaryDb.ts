import { existsSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";
import { join } from "node:path";

import summaryArtifact from "../data/ai-summaries.json";

export interface LocalAiSummary {
  content: string;
  model: string | null;
  sourceHash: string | null;
}

function databasePath(): string {
  return process.env.AI_SUMMARY_DB_PATH?.trim() || join(process.cwd(), ".hyacine", "data.db");
}

function projectPathCandidates(postId: string): string[] {
  const normalized = postId.replaceAll("\\", "/").replace(/^\/+/, "");
  const withExtension = /\.(?:md|mdx)$/i.test(normalized)
    ? [normalized]
    : [`${normalized}.md`, `${normalized}.mdx`];
  return [
    normalized,
    `src/posts/${normalized}`,
    `@/src/posts/${normalized}`,
    ...withExtension.flatMap((path) => [`src/posts/${path}`, `@/src/posts/${path}`]),
  ];
}

/** 读取随源码部署的摘要构建产物，供 Cloudflare Pages 等无本地数据库环境使用。 */
function readSummaryArtifact(postId: string, postTitle?: string): LocalAiSummary | null {
  const candidates = new Set(projectPathCandidates(postId).map((path) => path.toLowerCase()));
  const title = postTitle?.trim() || "";
  const entry = summaryArtifact.entries.find(
    (candidate) =>
      candidates.has(candidate.path.toLowerCase()) || (title !== "" && candidate.title === title),
  );
  const content = entry?.content.trim() || "";
  if (!content) return null;
  return {
    content,
    model: entry?.model?.trim() || null,
    sourceHash: null,
  };
}

/** 从本地摘要数据库或静态构建产物读取一篇文章的摘要。数据缺失时安全返回 null。 */
export function readLocalAiSummary(postId: string, postTitle?: string): LocalAiSummary | null {
  if (!postId) return null;
  if (!existsSync(databasePath())) return readSummaryArtifact(postId, postTitle);

  let database: DatabaseSync | undefined;
  try {
    database = new DatabaseSync(databasePath(), { readOnly: true });
    const paths = projectPathCandidates(postId);
    // Astro normalizes collection ids to lowercase for route params, while the
    // Hyacine Post table preserves the original filename casing.
    const normalizedPaths = paths.map((path) => path.toLowerCase());
    const placeholders = normalizedPaths.map(() => "?").join(", ");
    const normalizedTitle = postTitle?.trim() || "";
    const titleClause = normalizedTitle ? " OR title = ?" : "";
    const row = database
      .prepare(
        `SELECT summary, summaryModel, summarySourceHash
         FROM Post
         WHERE lower(path) IN (${placeholders})${titleClause}
         LIMIT 1`,
      )
      .get(...normalizedPaths, ...(normalizedTitle ? [normalizedTitle] : [])) as
      | { summary?: unknown; summaryModel?: unknown; summarySourceHash?: unknown }
      | undefined;
    const content = typeof row?.summary === "string" ? row.summary.trim() : "";
    if (!content) return readSummaryArtifact(postId, postTitle);
    return {
      content,
      model:
        typeof row?.summaryModel === "string" && row.summaryModel.trim()
          ? row.summaryModel.trim()
          : null,
      sourceHash: typeof row?.summarySourceHash === "string" ? row.summarySourceHash : null,
    };
  } catch {
    // 兼容早期本地脚本创建的 ai_summaries 表；正式格式与旧版 Hyacine 一致，使用 Post 表。
    try {
      const paths = projectPathCandidates(postId);
      const normalizedPaths = paths.map((path) => path.toLowerCase());
      const row = database
        ?.prepare(
          `SELECT summary, model, source_hash AS sourceHash
           FROM ai_summaries
           WHERE lower(post_id) IN (${normalizedPaths.map(() => "?").join(", ")})
           LIMIT 1`,
        )
        .get(...normalizedPaths) as
        | { summary?: unknown; model?: unknown; sourceHash?: unknown }
        | undefined;
      const content = typeof row?.summary === "string" ? row.summary.trim() : "";
      if (!content) return readSummaryArtifact(postId, postTitle);
      return {
        content,
        model: typeof row?.model === "string" && row.model.trim() ? row.model.trim() : null,
        sourceHash: typeof row?.sourceHash === "string" ? row.sourceHash : null,
      };
    } catch {
      return readSummaryArtifact(postId, postTitle);
    }
  } finally {
    database?.close();
  }
}
