import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import { DatabaseSync } from "node:sqlite";

const root = process.cwd();
const dbPath = process.env.AI_SUMMARY_DB_PATH?.trim() || join(root, ".hyacine", "data.db");
const outputPath = join(root, "src", "data", "ai-summaries.json");

if (!existsSync(dbPath)) {
  throw new Error(`找不到摘要数据库：${dbPath}`);
}

const database = new DatabaseSync(dbPath, { readOnly: true });
try {
  const rows = database
    .prepare(
      `SELECT path, title, summary, summaryModel
       FROM Post
       WHERE typeof(summary) = 'text' AND trim(summary) <> ''
       ORDER BY path`,
    )
    .all();
  const entries = rows.map((row) => ({
    path: typeof row.path === "string" ? row.path : "",
    title: typeof row.title === "string" ? row.title : "",
    content: typeof row.summary === "string" ? row.summary.trim() : "",
    model: typeof row.summaryModel === "string" ? row.summaryModel.trim() : null,
  }));

  mkdirSync(join(root, "src", "data"), { recursive: true });
  writeFileSync(outputPath, `${JSON.stringify({ version: 1, entries }, null, 2)}\n`, "utf8");
  console.log(`已导出 ${entries.length} 条 AI 摘要：${relative(root, outputPath)}`);
} finally {
  database.close();
}
