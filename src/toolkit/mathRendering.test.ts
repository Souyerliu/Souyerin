// @vitest-environment jsdom

import { markdownToHtml, mdxToJs } from "satteri";
import { describe, expect, it } from "vitest";

import katex from "../satteri-plugins/katex";

describe("math 代码围栏渲染", () => {
  it("Markdown 围栏输出真实公式，普通代码保持原样", async () => {
    const source =
      "```math\n\\begin{aligned}x&=1\\\\y&=2\\end{aligned}\n```\n\n```js\nconst x = 1;\n```";
    const { html } = await markdownToHtml(source, {
      features: { math: true },
      mdastPlugins: [katex({ throwOnError: true })],
    });
    const document = new DOMParser().parseFromString(html, "text/html");
    expect(document.querySelectorAll(".katex-display")).toHaveLength(1);
    expect(document.querySelector("code.language-math")).toBeNull();
    expect(document.querySelector("code.language-js")?.textContent).toContain("const x = 1;");
  });

  it("折叠块和 Tabs 内的围栏使用 MDX Katex 组件，保留普通代码", async () => {
    const source =
      '<Tabs>\n<Tab label="答案">\n<Collapse title="点击查看答案">\n\n```math\nx^2\n```\n\n```js\nconst x = 1;\n```\n\n</Collapse>\n</Tab>\n</Tabs>';
    const { code } = await mdxToJs(source, {
      features: { math: true },
      mdastPlugins: [katex()],
    });
    expect(code).toContain("Katex");
    expect(code).toContain("displayMode");
    expect(code).not.toContain("language-math");
    expect(code).not.toContain("katex-html");
    expect(code).toContain("language-js");
  });
});
