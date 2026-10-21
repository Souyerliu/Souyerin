import { readdir, readFile, writeFile } from "node:fs/promises";
import { gzipSync } from "node:zlib";

/* oxlint-disable no-await-in-loop -- 按模型顺序压缩，限制大型二进制副本的内存占用。 */

// 原始核心保留用于回退；每次开发/构建更新压缩副本，避免模型更新后使用旧资源。
const root = new URL("../public/live2d-models/model/yuno/", import.meta.url);
for (const directory of await readdir(root, { withFileTypes: true })) {
  if (!directory.isDirectory()) continue;
  const modelRoot = new URL(`${directory.name}/`, root);
  for (const file of await readdir(modelRoot)) {
    if (!file.endsWith(".moc3")) continue;
    const source = await readFile(new URL(file, modelRoot));
    const compressed = gzipSync(source, { level: 9 });
    const output = new URL(`${file}.gz`, modelRoot);
    const previous = await readFile(output).catch(() => null);
    if (!previous?.equals(compressed)) await writeFile(output, compressed);
    console.log(`[Live2D] ${file}: ${source.length} → ${compressed.length} bytes`);
  }
}
