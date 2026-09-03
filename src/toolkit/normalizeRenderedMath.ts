/**
 * Satteri 的 Markdown raw HTML 兼容路径可能把 KaTeX 包成嵌套段落：
 *   <p>正文 <p><span class="katex">...</span></p></p>
 *
 * 只移除包住单个 KaTeX 根节点的那一层段落，避免让浏览器修正无效 HTML
 * 时改变公式位置；公式内部的 span、MathML 和 SVG 不做任何字符串改写。
 */
const DISPLAY_MATH_PARAGRAPH = /<p>(?<math><span class="katex-display">[\s\S]*?)<\/p>/g;
const INLINE_MATH_PARAGRAPH = /<p>(?<math><span class="katex">[\s\S]*?)<\/p>/g;

export function normalizeRenderedMath(html: string): string {
  return html
    .replace(DISPLAY_MATH_PARAGRAPH, "$<math>$")
    .replace(INLINE_MATH_PARAGRAPH, "$<math>$");
}
