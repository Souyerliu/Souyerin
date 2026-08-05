// 分类封面图片优化脚本：将 public/images 下 PNG 封面转换为 WebP 并删除原文件
// 用法: bun run scripts/optimize-covers.mjs
import { existsSync, statSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import sharp from "sharp";

const dir = "public/images";

// 首页分类卡片封面（与 src/theme.config.ts 的 home.selectedCategories 对应）
const targets = ["NLP.png", "61B-header.png", "EECS127.png", "ml.png"];

for (const file of targets) {
  const input = join(dir, file);
  if (!existsSync(input)) {
    console.warn(`跳过（不存在）: ${file}`);
    continue;
  }
  const before = statSync(input).size;
  const output = join(dir, file.replace(/\.png$/i, ".webp"));
  // 分类卡片在桌面端宽度约 600px，限制 1200px（覆盖 2x 高分屏）即可
  const info = await sharp(input)
    .resize({ width: 1200, withoutEnlargement: true })
    .webp({ quality: 82 })
    .toFile(output);
  const after = statSync(output).size;
  unlinkSync(input);
  console.log(
    `${file}: ${(before / 1024).toFixed(1)}KB -> ${info.width}x${info.height} WebP ${(after / 1024).toFixed(1)}KB (节省 ${(100 - (after / before) * 100).toFixed(0)}%)`,
  );
}

console.log("完成。请同步更新 src/theme.config.ts 中对应的 .webp 引用。");
