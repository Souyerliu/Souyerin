/**
 * Satteri 会正确输出 GFM 表格，但不会自动添加主题使用的滚动容器。
 * 在文章最终 HTML 中补上容器，让表格获得主题样式并在窄屏下可横向滚动。
 */
const TABLE_PATTERN =
  /<div class="table-container">(?<wrapped><table\b[\s\S]*?<\/table>)<\/div>|(?<table><table\b[\s\S]*?<\/table>)/g;

export function wrapRenderedTables(html: string): string {
  return html.replace(TABLE_PATTERN, (match, wrapped, table) => {
    if (wrapped) return match;
    return `<div class="table-container">${table}</div>`;
  });
}
