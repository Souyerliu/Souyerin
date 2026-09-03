import { createHash } from "node:crypto";
import { mkdir, readdir, readFile, stat } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { existsSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { DatabaseSync } from "node:sqlite";

const root = process.cwd();
const postsDir = join(root, "src", "posts");
const force = process.argv.includes("--force");

function loadDotEnv() {
  const envPath = join(root, ".env");
  if (!existsSync(envPath)) return;
  const text = readFileSync(envPath, "utf8");
  for (const line of text.split(/\r?\n/)) {
    const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (!match || match[1] in process.env) continue;
    const value = match[2].replace(/^(["'])(.*)\1$/, "$2");
    process.env[match[1]] = value;
  }
}

function readConfig(name, fallback = "") {
  return (process.env[name] ?? fallback).trim();
}

function numberConfig(name, fallback) {
  const value = Number(readConfig(name, String(fallback)));
  return Number.isFinite(value) ? value : fallback;
}

function isPostFile(name) {
  return /\.(?:md|mdx)$/i.test(name);
}

async function collectPosts(directory, prefix = "") {
  const entries = await readdir(directory, { withFileTypes: true });
  const paths = [];
  for (const entry of entries) {
    if (entry.name.startsWith("_") || entry.name === ".git") continue;
    const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
    const fullPath = join(directory, entry.name);
    // Keep traversal deterministic and avoid loading every directory at once.
    // oxlint-disable-next-line no-await-in-loop -- recursive traversal is intentionally sequential.
    if (entry.isDirectory()) paths.push(...(await collectPosts(fullPath, relativePath)));
    else if (entry.isFile() && isPostFile(entry.name)) paths.push({ fullPath, relativePath });
  }
  return paths;
}

function isEncrypted(markdown) {
  const frontmatter = markdown.match(/^---\r?\n([\s\S]*?)\r?\n---/i)?.[1] ?? "";
  return /(?:^|\n)\s*encrypted\s*:\s*true\s*(?:#.*)?(?=\n|$)/i.test(frontmatter);
}

function readTitle(markdown, relativePath) {
  const frontmatter = markdown.match(/^---\r?\n([\s\S]*?)\r?\n---/i)?.[1] ?? "";
  const title = frontmatter.match(/(?:^|\n)\s*title\s*:\s*(.+?)\s*(?:\n|$)/i)?.[1]?.trim();
  return title?.replace(/^(["'])(.*)\1$/, "$2") || relativePath.replace(/\.(?:md|mdx)$/i, "");
}

function stripFrontmatter(markdown) {
  const match = markdown.match(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/i);
  return (match ? markdown.slice(match[0].length) : markdown).trim();
}

function normalizeSummary(value) {
  return value.replace(/\s+/g, " ").trim();
}

function toProjectPath(relativePath) {
  return `@/src/posts/${relativePath.replaceAll("\\", "/")}`;
}

function toSlug(relativePath) {
  return (
    relativePath
      .replace(/\.(?:md|mdx)$/i, "")
      .replaceAll("\\", "/")
      .replaceAll("/", "-")
      .replace(/[^\p{L}\p{N}-]+/gu, "-")
      .replace(/-+/g, "-")
      .replace(/^-+|-+$/g, "") || "post"
  );
}

async function requestSummary(markdown, config) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000);
  try {
    const response = await fetch(`${config.apiBaseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${config.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: config.model,
        temperature: config.temperature,
        max_tokens: config.maxOutputTokens,
        messages: [
          {
            role: "system",
            content:
              "你是一个专业中文技术编辑。请对输入文章生成精炼概括，输出 1 段中文，长度约 80-180 字，避免项目符号和多余前缀。",
          },
          {
            role: "user",
            content: `请概括下面文章内容：\n\n${stripFrontmatter(markdown).slice(0, config.maxInputChars)}`,
          },
        ],
      }),
      signal: controller.signal,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(
        `HTTP ${response.status}: ${payload?.error?.message || "summary request failed"}`,
      );
    }
    const rawContent = payload?.choices?.[0]?.message?.content;
    const content =
      typeof rawContent === "string"
        ? normalizeSummary(rawContent)
        : normalizeSummary(
            Array.isArray(rawContent)
              ? rawContent
                  .filter((item) => item?.type === "text")
                  .map((item) => item?.text || "")
                  .join(" ")
              : "",
          );
    if (!content) throw new Error("summary response is empty");
    return content;
  } finally {
    clearTimeout(timeout);
  }
}

loadDotEnv();

const dbPath = process.env.AI_SUMMARY_DB_PATH?.trim() || join(root, ".hyacine", "data.db");
const config = {
  apiBaseUrl: readConfig("AI_SUMMARY_API_BASE_URL").replace(/\/+$/, ""),
  apiKey: readConfig("AI_SUMMARY_API_KEY"),
  model: readConfig("AI_SUMMARY_MODEL"),
  maxInputChars: Math.max(256, Math.floor(numberConfig("AI_SUMMARY_MAX_INPUT_CHARS", 50_000))),
  maxOutputTokens: Math.max(1, Math.floor(numberConfig("AI_SUMMARY_MAX_OUTPUT_TOKENS", 256))),
  temperature: Math.min(2, Math.max(0, numberConfig("AI_SUMMARY_TEMPERATURE", 0.2))),
};

if (!config.apiBaseUrl || !config.apiKey || !config.model) {
  throw new Error("缺少 AI_SUMMARY_API_BASE_URL、AI_SUMMARY_API_KEY 或 AI_SUMMARY_MODEL 配置");
}

const posts = await collectPosts(postsDir);
await mkdir(dirname(dbPath), { recursive: true });
const database = new DatabaseSync(dbPath);
database.exec(`
  CREATE TABLE IF NOT EXISTS Post (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    draft BOOLEAN NOT NULL DEFAULT 0,
    categories JSONB NOT NULL,
    hash TEXT NOT NULL,
    createdAt DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updatedAt DATETIME NOT NULL,
    lastModified DATETIME NOT NULL,
    summary TEXT,
    summaryError TEXT,
    summaryModel TEXT,
    summarySourceHash TEXT,
    summaryUpdatedAt DATETIME
  );
`);
const postColumns = new Set(
  database
    .prepare("PRAGMA table_info(Post)")
    .all()
    .map((column) => column.name),
);
for (const [name, definition] of [
  ["summary", "TEXT"],
  ["summaryError", "TEXT"],
  ["summaryModel", "TEXT"],
  ["summarySourceHash", "TEXT"],
  ["summaryUpdatedAt", "DATETIME"],
]) {
  if (!postColumns.has(name)) database.exec(`ALTER TABLE Post ADD COLUMN ${name} ${definition}`);
}
database.exec("CREATE INDEX IF NOT EXISTS Post_summaryUpdatedAt_idx ON Post(summaryUpdatedAt)");
const findExisting = database.prepare(
  "SELECT summary, summaryModel, summarySourceHash, summaryUpdatedAt, lastModified FROM Post WHERE path = ?",
);
const upsert = database.prepare(`
  INSERT INTO Post
    (path, title, slug, draft, categories, hash, createdAt, updatedAt, lastModified,
     summary, summaryError, summaryModel, summarySourceHash, summaryUpdatedAt)
  VALUES (?, ?, ?, 0, '[]', ?, ?, ?, ?, ?, ?, ?, ?, ?)
  ON CONFLICT(path) DO UPDATE SET
    path = excluded.path,
    title = excluded.title,
    hash = excluded.hash,
    updatedAt = excluded.updatedAt,
    lastModified = excluded.lastModified,
    summary = excluded.summary,
    summaryError = excluded.summaryError,
    summaryModel = excluded.summaryModel,
    summarySourceHash = excluded.summarySourceHash,
    summaryUpdatedAt = excluded.summaryUpdatedAt
`);

let generated = 0;
let skipped = 0;
let failed = 0;
for (const post of posts) {
  // Process one article at a time to avoid flooding the provider and to keep DB writes ordered.
  // oxlint-disable-next-line no-await-in-loop -- intentional sequential generation.
  const markdown = await readFile(post.fullPath, "utf8");
  // 与 Hyacine Post.lastModified 保持一致：记录源文章文件的真实修改时间，
  // 而不是摘要生成时间。
  // oxlint-disable-next-line no-await-in-loop -- files are intentionally processed sequentially.
  const fileStat = await stat(post.fullPath);
  const lastModified = new Date(fileStat.mtimeMs).toISOString();
  if (isEncrypted(markdown)) {
    skipped += 1;
    continue;
  }
  const postId = post.relativePath.replaceAll("\\", "/");
  const projectPath = toProjectPath(postId);
  const title = readTitle(markdown, post.relativePath);
  const slug = toSlug(postId);
  const sourceHash = createHash("sha256").update(markdown).digest("hex");
  const existing = findExisting.get(projectPath);
  if (
    !force &&
    existing?.summarySourceHash === sourceHash &&
    typeof existing.summary === "string" &&
    existing.summary.trim()
  ) {
    // 摘要无需重算时，仍修正/刷新数据库中的文件修改时间。
    if (existing.lastModified !== lastModified) {
      database
        .prepare("UPDATE Post SET updatedAt = ?, lastModified = ? WHERE path = ?")
        .run(new Date().toISOString(), lastModified, projectPath);
    }
    skipped += 1;
    continue;
  }

  try {
    // oxlint-disable-next-line no-await-in-loop -- intentional sequential generation.
    const summary = await requestSummary(markdown, config);
    const now = new Date().toISOString();
    upsert.run(
      projectPath,
      title,
      slug,
      sourceHash,
      now,
      now,
      lastModified,
      summary,
      null,
      config.model,
      sourceHash,
      now,
    );
    generated += 1;
    console.log(`已生成: ${postId}`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const now = new Date().toISOString();
    upsert.run(
      projectPath,
      title,
      slug,
      sourceHash,
      now,
      now,
      lastModified,
      existing?.summary || null,
      message,
      config.model,
      existing?.summarySourceHash || null,
      existing?.summaryUpdatedAt || null,
    );
    failed += 1;
    console.error(`生成失败: ${postId} (${message})`);
  }
}

database.close();
console.log(
  `摘要处理完成：生成 ${generated}，跳过 ${skipped}，失败 ${failed}，数据库 ${relative(root, dbPath)}`,
);
if (failed > 0) process.exitCode = 1;
