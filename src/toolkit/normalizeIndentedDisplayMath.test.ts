import { describe, expect, it } from "vitest";

import {
  normalizeIndentedDisplayMath,
  normalizeIndentedMdxDisplayMath,
} from "./normalizeIndentedDisplayMath";

describe("normalizeIndentedDisplayMath", () => {
  it("normalizes indented display-math fences", () => {
    expect(normalizeIndentedDisplayMath("- item\n    $$\n    x^2\n    $$")).toBe(
      "- item\n    ```math\n    x^2\n    ```",
    );
  });

  it("leaves dollar fences inside fenced code blocks untouched", () => {
    const markdown = "```md\n    $$\n    x^2\n    $$\n```";
    expect(normalizeIndentedDisplayMath(markdown)).toBe(markdown);
  });

  it("preserves CRLF line endings", () => {
    expect(normalizeIndentedDisplayMath("- item\r\n    $$\r\n    x^2\r\n    $$")).toBe(
      "- item\r\n    ```math\r\n    x^2\r\n    ```",
    );
  });

  it("保留顶层公式、未闭合公式与波浪号代码围栏", () => {
    for (const source of ["$$\nx^2\n  $$", "  $$\nx^2", "~~~md\n  $$\n  x^2\n  $$\n~~~"]) {
      expect(normalizeIndentedDisplayMath(source)).toBe(source);
    }
  });

  it("再次处理不会改变公式代码围栏", () => {
    const normalized = normalizeIndentedDisplayMath("- item\n  $$\n  x^2\n  $$\n  after");
    expect(normalizeIndentedDisplayMath(normalized)).toBe(normalized);
  });
});

describe("normalizeIndentedMdxDisplayMath", () => {
  it("converts indented math to the existing Katex component", () => {
    expect(normalizeIndentedMdxDisplayMath("- item\n    $$\n    x^2\n    $$")).toBe(
      '- item\n   <Katex valueBase64="eF4y" displayMode />',
    );
  });

  it("leaves top-level and fenced-code dollar blocks untouched", () => {
    const markdown = "$$\nx^2\n$$\n```md\n    $$\n    y^2\n    $$\n```";
    expect(normalizeIndentedMdxDisplayMath(markdown)).toBe(markdown);
  });

  it("supports a previously normalized closing fence", () => {
    expect(normalizeIndentedMdxDisplayMath("    $$\n    x^2\n$$")).toBe(
      '   <Katex valueBase64="eF4y" displayMode />',
    );
  });

  it("does not reinterpret an indented closing fence", () => {
    const markdown = "$$\nx^2\n    $$";
    expect(normalizeIndentedMdxDisplayMath(markdown)).toBe(markdown);
  });

  it("converts a display formula used as a list item", () => {
    expect(normalizeIndentedMdxDisplayMath("    + $$\n        x^2\n      $$")).toBe(
      '    + <Katex valueBase64="eF4y" displayMode />',
    );
  });

  it("supports formula content on the opening line", () => {
    expect(normalizeIndentedMdxDisplayMath("  $$x^2\n  $$")).toBe(
      ' <Katex valueBase64="eF4y" displayMode />',
    );
  });
});
