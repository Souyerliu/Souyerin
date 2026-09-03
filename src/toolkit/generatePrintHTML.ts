/**
 * generatePrintHTML.ts — 为文章生成独立的打印页面 HTML
 *
 * 方案：fetch 页面所有 stylesheet 的原始 CSS 文本，完整注入打印页。
 * 浏览器原生 window.print() 支持 oklch() 等现代颜色函数，无兼容问题。
 */

/**
 * 获取当前页面所有 stylesheet 的 CSS 文本。
 * 对同源 <link> 标签用 fetch 获取内容；<style> 标签直接取 textContent。
 * 所有 url() 中的相对/绝对路径统一转为 origin 前缀的绝对 URL，
 * 确保 blob: 协议下字体等资源能正确加载。
 */
export async function fetchAllStylesheets(): Promise<string> {
  const origin = window.location.origin;
  const parts: string[] = [];

  // <style> 标签
  document.querySelectorAll("style").forEach((el) => {
    if (el.textContent) parts.push(el.textContent);
  });

  // <link rel="stylesheet"> 标签 — 同源 fetch
  const links = document.querySelectorAll<HTMLLinkElement>('link[rel="stylesheet"]');
  const results = await Promise.allSettled(
    Array.from(links).map(async (link) => {
      const href = link.href;
      if (!href.startsWith(origin)) return "";
      try {
        const res = await fetch(href);
        if (!res.ok) return "";
        const cssText = await res.text();
        // 以当前 CSS 文件的 href 为基准，将 url() 中的相对/站点绝对路径转为全限定 URL
        const base = href.substring(0, href.lastIndexOf("/") + 1);
        return cssText.replace(
          /url\((["']?)((?:\.\.?\/[^)"']+|\/[^)"']+)[^)"']*)\1\)/g,
          (_: string, q: string, path: string) => {
            const full = new URL(path.startsWith("/") ? path : base + path, origin).href;
            return `url(${q}${full}${q})`;
          },
        );
      } catch {
        return "";
      }
    }),
  );
  for (const r of results) {
    if (r.status === "fulfilled" && r.value) parts.push(r.value);
  }

  // <style> 中的相对/绝对 url() 也做相同转换（以页面 origin 为基准）
  return parts
    .join("\n")
    .replace(
      /url\((["']?)((?:\.\.?\/[^)"']+|\/[^)"']+)[^)"']*)\1\)/g,
      (_: string, q: string, path: string) => `url(${q}${new URL(path, origin).href}${q})`,
    );
}

/** 将克隆 DOM 中 <img> 的相对路径和站点绝对路径转为完整 URL（blob 协议下也能加载） */
export function resolveImageUrls(clone: HTMLElement): void {
  const origin = window.location.origin;
  clone.querySelectorAll("img").forEach((el) => {
    const raw = el.getAttribute("src");
    if (!raw || raw.startsWith("data:") || raw.startsWith("http")) return;
    try {
      // 相对路径和 / 开头的站点绝对路径都转为完整 URL
      el.src = new URL(raw, origin).href;
    } catch {
      /* 忽略 */
    }
  });
}

/**
 * 极简样式内联：仅把文字颜色、背景色和 KaTeX 依赖的关键排版属性写入内联。
 * 不碰 border/outline，避免产生黑框。
 * **跳过所有 .katex 子孙元素**，避免破坏 KaTeX 内部精细的绝对定位/间距，
 * 从而修复 $\neq$ 等符号的偏移问题。
 */
export function inlineEssentialStyles(original: HTMLElement, clone: HTMLElement): void {
  const ow = document.createTreeWalker(original, NodeFilter.SHOW_ELEMENT);
  const cw = document.createTreeWalker(clone, NodeFilter.SHOW_ELEMENT);

  const props: Array<{ name: string; skip?: string }> = [
    { name: "color" },
    { name: "background-color", skip: "rgba(0, 0, 0, 0)" },
    { name: "font-size" },
    { name: "display", skip: "inline" },
  ];

  while (ow.nextNode() && cw.nextNode()) {
    const origEl = ow.currentNode;
    const cloneEl = cw.currentNode;
    if (!(origEl instanceof HTMLElement) || !(cloneEl instanceof HTMLElement)) continue;

    // 不碰 KaTeX 内部元素，避免破坏符号渲染
    if (origEl.closest(".katex")) continue;

    const cs = getComputedStyle(origEl);

    for (const { name, skip } of props) {
      const val = cs.getPropertyValue(name);
      if (!val || val === "transparent" || val === skip) continue;
      cloneEl.style.setProperty(name, val);
    }
  }
}

/**
 * 构建打印页面完整 HTML。
 * - 内联所有页面 CSS，KaTeX 字体 path 自动正确
 * - 颜色用 computed rgb() 内联，不存在 oklch 解析问题
 */
export function buildPrintHTML(title: string, clone: HTMLElement, allCSS: string): string {
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>${title}</title>
<style>
${allCSS}

/* 打印分页 — A4 纸宽 210mm，留 12mm 页边距 = 186mm 可用宽度 */
@page { size: A4; margin: 12mm; }
@media print {
  /* 强制全页级联匹配 A4 宽度，覆盖所有主题容器固定宽度 */
  html, body, #container, main, .wrap, .article, .post, .block, .post.block,
  .md, .body, .content, article, section, div {
    width: 100% !important;
    max-width: none !important;
    min-width: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
  }
  body {
    background: #fff;
    font-size: 13.5px;
    line-height: 1.6;
  }

  /* 分页控制：只在真正需要的地方避免内部分页，其余段落/列表允许自然断页 */
  h2, h3, h4 { break-before: avoid; break-after: avoid; }
  .post-header { break-after: avoid; }
  /* 不可分割块 */
  pre, blockquote, table, img { break-inside: avoid; }
  .katex-display, .katex, .katex-inline { break-inside: avoid; }
  /* 段落和列表不做限制，允许跨页以消除空白 */

  /* 公式：!important 保持不变用于对抗 KaTeX 内联样式 */
  .katex-display {
    break-inside: avoid;
    max-width: 100%;
  }
  .katex-display > .katex {
    max-width: 100%;
  }
  .katex {
    break-inside: avoid;
    max-width: 100%;
  }

  /* 网格和代码块 */
  table { max-width: 100%; word-break: break-word; }
  pre { max-width: 100%; overflow-x: auto; white-space: pre-wrap; word-break: break-word; }
  img { max-width: 100%; height: auto; }
}
</style>
</head>
<body>${clone.outerHTML}</body>
</html>`;
}
