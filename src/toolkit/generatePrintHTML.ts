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

/** A4 纸宽 210mm 减去左右各 12mm 页边距后的可用版心宽度 */
export const PRINT_CONTENT_WIDTH = "186mm";

/**
 * 把超出版心宽度的行间公式按比例缩小到恰好放得下。
 *
 * 以 `.katex-display` 自身的 `clientWidth`（版心可用宽度）与 `scrollWidth`（内容真实宽度）
 * 作比例基准，比用 body 宽度推算更准；缩放通过百分比 font-size 施加，
 * 这样 KaTeX 内部所有 em 尺寸都会等比跟随。
 *
 * 先统一测量再统一写入，避免逐个改写触发的重复回流；
 * 最后把 `overflow` 置为 visible，消除预览页里的公式横向滚动条。
 *
 * @param root 打印页文档（或任意容器）
 * @param minScale 缩放下限，防止超长公式被压到不可读
 */
export function fitDisplayMath(root: ParentNode, minScale = 0.45): void {
  const displays = Array.from(root.querySelectorAll<HTMLElement>(".katex-display"));

  const scales = displays.map((display) => {
    const available = display.clientWidth;
    // 内层 .katex 被 max-width 夹住时 offsetWidth 不可信，取两者 scrollWidth 的较大值
    const inner = display.querySelector<HTMLElement>(":scope > .katex");
    const real = Math.max(display.scrollWidth, inner?.scrollWidth ?? 0);
    // 允许 1px 的测量误差，避免无谓缩放
    if (!available || !real || real <= available + 1) return 1;
    return Math.max(minScale, available / real);
  });

  displays.forEach((display, index) => {
    const scale = scales[index] ?? 1;
    if (scale < 1) {
      display.style.setProperty("font-size", `${(scale * 100).toFixed(2)}%`, "important");
    }
    display.style.setProperty("overflow", "visible", "important");
  });
}

/**
 * 极简样式内联：仅把文字颜色、背景色和 KaTeX 依赖的关键排版属性写入内联。
 * 不碰 border/outline，避免产生黑框。
 * **跳过所有 .katex / .katex-display 元素**，避免破坏 KaTeX 内部精细的绝对定位/间距
 * （并保证行间公式的字号可以被后续缩放逻辑接管）。
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

    // 不碰 KaTeX 元素，避免破坏符号渲染、并留出公式缩放空间
    if (origEl.closest(".katex, .katex-display")) continue;

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
 * - 屏幕预览与打印共用同一套 A4 版心宽度，保证所见即所得
 */
export function buildPrintHTML(title: string, clone: HTMLElement, allCSS: string): string {
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>${title}</title>
<style>
${allCSS}

/* 打印分页 — A4 纸宽 210mm，留 12mm 页边距 = ${PRINT_CONTENT_WIDTH} 可用宽度 */
@page { size: A4; margin: 12mm; }

/* 布局：屏幕预览与打印共用，避免预览尺寸与实际出纸不一致 */
html, body, #container, main, .wrap, .article, .post, .block, .post.block,
.md, .body, .content, .post-content, article {
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

/* 屏幕预览时模拟 A4 版心，宽度与打印一致且不产生横向滚动条 */
@media screen {
  html { overflow-x: hidden; }
  body {
    box-sizing: border-box;
    width: ${PRINT_CONTENT_WIDTH} !important;
    max-width: 100% !important;
    margin: 0 auto !important;
  }
}

/* 溢出控制 */
table { max-width: 100%; word-break: break-word; }
pre { max-width: 100%; white-space: pre-wrap; word-break: break-word; }
img { max-width: 100%; height: auto; }

/* 公式：宽度受版心约束，超宽部分由 fitDisplayMath 等比缩小 */
.katex-display,
.katex-display > .katex,
.katex {
  max-width: 100%;
}

@media print {
  /* 分页控制：只在真正需要的地方避免内部分页，其余段落/列表允许自然断页 */
  h2, h3, h4 { break-before: avoid; break-after: avoid; }
  .post-header { break-after: avoid; }
  /* 不可分割块 */
  pre, blockquote, table, img { break-inside: avoid; }
  .katex-display, .katex, .katex-inline { break-inside: avoid; }
  pre { overflow-x: hidden; }
}
</style>
</head>
<body>${clone.outerHTML}</body>
</html>`;
}
