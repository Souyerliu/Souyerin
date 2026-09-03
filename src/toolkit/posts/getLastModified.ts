/**
 * 获取文章源文件的最后修改时间。
 * 优先读取本地 Hyacine Post.lastModified，缺失时回退到文件系统 mtime。
 */
import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

function findProjectRoot(): string {
  let directory = process.cwd();
  for (let index = 0; index < 10; index += 1) {
    if (fs.existsSync(path.join(directory, "astro.config.mjs"))) return directory;
    const parent = path.dirname(directory);
    if (parent === directory) break;
    directory = parent;
  }
  return process.cwd();
}

const PROJECT_ROOT = findProjectRoot();
const POSTS_BASE = path.resolve(PROJECT_ROOT, "src/posts");

function databasePath(): string {
  return (
    process.env.AI_SUMMARY_DB_PATH?.trim() ||
    path.resolve(PROJECT_ROOT, ".hyacine/data.db")
  );
}

/** 将文件路径转换为 Astro 内容集合使用的 ID。 */
function toAstroId(relativePath: string): string {
  const withoutExtension = relativePath.replace(/\.(mdx|md)$/i, "");
  return withoutExtension
    .toLowerCase()
    .replace(/\\/g, "/")
    .replace(/\./g, "")
    .replace(/[\uff08\uff09\u3000-\u303f\ufe30-\ufe4f]/g, "")
    .replace(/[^a-z0-9/\u4e00-\u9fff\u3400-\u4dbf]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

function buildSlugMap(): Map<string, string> {
  const map = new Map<string, string>();
  const pending = [POSTS_BASE];

  while (pending.length > 0) {
    const directory = pending.pop();
    if (!directory) continue;

    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(directory, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      const absolutePath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        pending.push(absolutePath);
      } else if (/\.(?:md|mdx)$/i.test(entry.name)) {
        const relativePath = path.relative(POSTS_BASE, absolutePath);
        map.set(toAstroId(relativePath), absolutePath);
      }
    }
  }

  return map;
}

const SLUG_MAP = buildSlugMap();

function parseDate(value: unknown): Date | null {
  if (typeof value !== "string" || !value.trim()) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function readDatabaseLastModified(postId: string, postTitle?: string): Date | null {
  const dbPath = databasePath();
  if (!fs.existsSync(dbPath)) return null;

  const normalized = postId.replaceAll("\\", "/").replace(/^\/+/, "");
  const basePaths = [normalized, `src/posts/${normalized}`, `@/src/posts/${normalized}`];
  const paths = [
    ...basePaths,
    ...basePaths.flatMap((candidate) => [`${candidate}.md`, `${candidate}.mdx`]),
  ].map((candidate) => candidate.toLowerCase());
  const placeholders = paths.map(() => "?").join(", ");
  const title = postTitle?.trim() || "";

  let database: DatabaseSync | undefined;
  try {
    database = new DatabaseSync(dbPath, { readOnly: true });
    const row = database
      .prepare(
        `SELECT lastModified
         FROM Post
         WHERE lower(path) IN (${placeholders})${title ? " OR title = ?" : ""}
         LIMIT 1`,
      )
      .get(...paths, ...(title ? [title] : [])) as { lastModified?: unknown } | undefined;
    return parseDate(row?.lastModified);
  } catch {
    return null;
  } finally {
    database?.close();
  }
}

export function getLastModified(
  postId: string,
  fallbackDate: Date,
  postTitle?: string,
): Date {
  return (
    readDatabaseLastModified(postId, postTitle) ||
    (() => {
      const realPath = SLUG_MAP.get(postId) || SLUG_MAP.get(toAstroId(postId));
      if (!realPath) return fallbackDate;
      try {
        return fs.statSync(realPath).mtime;
      } catch {
        return fallbackDate;
      }
    })()
  );
}
