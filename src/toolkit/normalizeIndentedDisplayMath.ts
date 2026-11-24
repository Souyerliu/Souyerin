/**
 * Satteri 无法可靠识别缩进的 $$。转换为现有 KaTeX 插件支持的 math
 * 代码围栏，保留列表缩进；把 $$ 移到行首会打断列表，让后续正文变成代码。
 * 普通代码围栏和顶层公式保持原样。
 */
export function normalizeIndentedDisplayMath(source: string): string {
  let codeFence: { marker: string; length: number } | undefined;
  let standardMath = false;
  const lines = source.split("\n");
  const output: string[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const content = line.endsWith("\r") ? line.slice(0, -1) : line;
    const carriageReturn = line.endsWith("\r") ? "\r" : "";
    const fence = content.match(/^\s*(`{3,}|~{3,})/);

    if (codeFence) {
      if (fence && fence[1][0] === codeFence.marker && fence[1].length >= codeFence.length) {
        codeFence = undefined;
      }
      output.push(line);
      continue;
    }

    if (fence) {
      codeFence = {
        marker: fence[1][0],
        length: fence[1].length,
      };
      output.push(line);
      continue;
    }

    if (standardMath) {
      output.push(line);
      if (/^\s*\$\$[\t ]*$/.test(content)) standardMath = false;
      continue;
    }
    if (/^\$\$(?!.*\$\$)/.test(content)) {
      standardMath = true;
      output.push(line);
      continue;
    }

    const opening = content.match(/^([\t ]+)\$\$[\t ]*$/);
    if (!opening) {
      output.push(line);
      continue;
    }
    let closingIndex = index + 1;
    while (closingIndex < lines.length && !/^\s*\$\$[\t ]*\r?$/.test(lines[closingIndex])) {
      closingIndex += 1;
    }
    if (closingIndex === lines.length) {
      output.push(line);
      continue;
    }

    const indentation = opening[1];
    // 使用比公式内容更长的围栏，避免公式中的反引号提前结束代码块。
    const body = lines.slice(index + 1, closingIndex);
    const marker = "`".repeat(
      Math.max(
        3,
        ...body.flatMap((value) => [...value.matchAll(/`+/g)].map((match) => match[0].length + 1)),
      ),
    );
    output.push(
      `${indentation}${marker}math${carriageReturn}`,
      ...body,
      `${indentation}${marker}${lines[closingIndex].endsWith("\r") ? "\r" : ""}`,
    );
    index = closingIndex;
  }
  return output.join("\n");
}

/**
 * MDX needs to retain list nesting around indented formulas. Converting the
 * formula to the existing Katex component avoids Satteri's indented `$$`
 * limitation without pulling the following list content out of its parent.
 */
export function normalizeIndentedMdxDisplayMath(source: string): string {
  const lines = source.split("\n");
  const output: string[] = [];
  let codeFence: { marker: string; length: number } | undefined;
  let standardMath = false;

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const content = line.endsWith("\r") ? line.slice(0, -1) : line;
    const fence = content.match(/^\s*(`{3,}|~{3,})/);

    if (codeFence) {
      output.push(line);
      if (fence && fence[1][0] === codeFence.marker && fence[1].length >= codeFence.length) {
        codeFence = undefined;
      }
      continue;
    }

    if (fence) {
      codeFence = {
        marker: fence[1][0],
        length: fence[1].length,
      };
      output.push(line);
      continue;
    }

    if (standardMath) {
      output.push(line);
      if (/^\s*\$\$[\t ]*$/.test(content)) standardMath = false;
      continue;
    }

    if (/^\$\$(?!.*\$\$)/.test(content)) {
      standardMath = true;
      output.push(line);
      continue;
    }

    const indentedOpening = content.match(/^([\t ]+)\$\$(.*)$/);
    const listOpening = content.match(/^(\s*(?:[-+*]|\d+[.)])\s+)\$\$(.*)$/);
    if (!indentedOpening && !listOpening) {
      output.push(line);
      continue;
    }

    let closingIndex = index + 1;
    while (closingIndex < lines.length && !/^\s*\$\$[\t ]*\r?$/.test(lines[closingIndex])) {
      closingIndex += 1;
    }

    if (closingIndex === lines.length) {
      output.push(line);
      continue;
    }

    const openingValue = (listOpening?.[2] ?? indentedOpening?.[2] ?? "").trim();
    const formulaLines = lines.slice(index + 1, closingIndex);
    const commonIndentation = Math.min(
      ...formulaLines
        .filter((formulaLine) => formulaLine.trim())
        .map((formulaLine) => formulaLine.match(/^[\t ]*/)?.[0].length ?? 0),
    );
    const bodyValue = formulaLines
      .map((formulaLine) => formulaLine.slice(commonIndentation))
      .join("\n")
      .trim();
    const value = [openingValue, bodyValue].filter(Boolean).join("\n");
    // Up to three leading spaces are valid at the document root. Removing one
    // also keeps the component nested when the formula belongs to a list item.
    const componentPrefix = listOpening ? listOpening[1] : indentedOpening![1].slice(1);
    const encodedValue = Buffer.from(value, "utf8").toString("base64");
    output.push(`${componentPrefix}<Katex valueBase64="${encodedValue}" displayMode />`);
    index = closingIndex;
  }

  return output.join("\n");
}
import { Buffer } from "node:buffer";
