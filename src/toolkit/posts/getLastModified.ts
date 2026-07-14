/**
 * 获取文章文件的最后修改时间
 * 优先从 hyc sync 数据库读取，回退到文件系统 mtime
 */
import fs from "node:fs";
import path from "node:path";

// 向上查找项目根目录（通过 astro.config.mjs 标记）
function findProjectRoot(): string {
  let dir = process.cwd();
  for (let i = 0; i < 10; i++) {
    if (fs.existsSync(path.join(dir, "astro.config.mjs"))) return dir;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return process.cwd();
}

const PROJECT_ROOT = findProjectRoot();
const POSTS_BASE = path.resolve(PROJECT_ROOT, "src/posts");
const DB_PATH = path.resolve(PROJECT_ROOT, ".hyacine/data.db");

/** 将文件路径转为 Astro 内容集合的 ID（slug） */
function toAstroId(relativePath: string): string {
  // 去掉扩展名 → 小写 → 反斜杠转正斜杠 → 去掉点号 → strip 全角标点（Astro 行为）→ 非字母数字中文转连字符 → 合并连续连字符
  const noExt = relativePath.replace(/\.(mdx|md)$/, "");
  return (
    noExt
      .toLowerCase()
      .replace(/\\/g, "/")
      .replace(/\./g, "")
      // 全角括号等标点 Astro 直接删除，不替换为连字符
      .replace(/[\uff08\uff09\u3000-\u303f\ufe30-\ufe4f]/g, "")
      .replace(/[^a-z0-9/\u4e00-\u9fff\u3400-\u4dbf]/g, "-")
      .replace(/-+/g, "-")
      .replace(/^-|-$/g, "")
  );
}

/** 递归扫描 posts 目录，构建 Astro ID → 绝对路径 的映射 */
function buildSlugMap(): Map<string, string> {
  const map = new Map<string, string>();
  const dirs = [POSTS_BASE];
  while (dirs.length > 0) {
    const dir = dirs.pop()!;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        dirs.push(fullPath);
      } else if (/\.(md|mdx)$/i.test(entry.name)) {
        const relativePath = path.relative(POSTS_BASE, fullPath);
        map.set(toAstroId(relativePath), fullPath);
      }
    }
  }
  return map;
}

const SLUG_MAP = buildSlugMap();
// 同时保留一个反向映射：绝对路径 → 相对路径（用于 DB LIKE 查询，正斜杠）
const PATH_TO_RELATIVE = new Map<string, string>();
for (const [, absPath] of SLUG_MAP) {
  PATH_TO_RELATIVE.set(absPath, path.relative(POSTS_BASE, absPath).replace(/\\/g, "/"));
}

let _dbPromise: Promise<any | null> | null = null;

function getDatabase(): Promise<any | null> {
  if (_dbPromise) return _dbPromise;
  _dbPromise = (async () => {
    try {
      if (!fs.existsSync(DB_PATH)) return null;
      const { Database } = await import("bun:sqlite");
      return new Database(DB_PATH);
    } catch {
      return null;
    }
  })();
  return _dbPromise;
}

export async function getLastModified(postId: string, fallbackDate: Date): Promise<Date> {
  // 1. 通过 slug 映射找到实际文件路径
  const realPath = SLUG_MAP.get(postId);

  if (realPath) {
    // 1a. 优先从 hyc sync 数据库读取
    const db = await getDatabase();
    if (db) {
      try {
        const relativePath = PATH_TO_RELATIVE.get(realPath);
        if (relativePath) {
          const pattern = `%/src/posts/${relativePath}`;
          const row = db
            .prepare("SELECT lastModified FROM Post WHERE path LIKE ?")
            .get(pattern) as { lastModified: string } | null;
          if (row?.lastModified) {
            const d = new Date(row.lastModified);
            if (!isNaN(d.getTime())) return d;
          }
        }
      } catch {
        /* 回退 */
      }
    }

    // 1b. 回退：直接读取文件系统 mtime
    try {
      return fs.statSync(realPath).mtime;
    } catch {
      /* 回退 */
    }
  }

  // 2. 最终回退到 frontmatter date
  return fallbackDate;
}
