import {
  defineMdastPlugin,
  type MdastPluginDefinition,
  type MdxJsxAttributeNode,
} from "satteri";

/**
 * katex（satteri 版）：
 * 将 math/inlineMath 节点转换为 Katex.astro 组件
 *
 * 不能直接返回 KaTeX HTML：Satteri 的 rawHtml 在 MDX 的 Astro Container
 * 路径中会被当作文本转义，最终页面会出现 &lt;span class="katex"...。
 * 交给 Astro 组件使用 set:html 注入，才能同时保留真实 DOM 和 KaTeX 的布局。
 * 需要 features.math: true 在 satteri 配置中开启解析
 */
export interface KatexOptions {
  /** KaTeX 渲染出错时是否抛出异常（默认 false，渲染为错误提示） */
  throwOnError?: boolean;
}

function valueAttribute(value: string): MdxJsxAttributeNode {
  return { type: "mdxJsxAttribute", name: "value", value };
}

function toKatexComponent(
  value: string,
  displayMode: boolean,
  throwOnError: boolean,
): {
  type: "mdxJsxFlowElement" | "mdxJsxTextElement";
  name: "Katex";
  attributes: MdxJsxAttributeNode[];
  children: [];
} {
  const attributes: MdxJsxAttributeNode[] = [valueAttribute(value)];
  if (displayMode) {
    attributes.push({
      type: "mdxJsxAttribute",
      name: "displayMode",
      value: null,
    });
  }
  if (throwOnError) {
    attributes.push({
      type: "mdxJsxAttribute",
      name: "throwOnError",
      value: null,
    });
  }

  return {
    type: displayMode ? "mdxJsxFlowElement" : "mdxJsxTextElement",
    name: "Katex",
    attributes,
    children: [],
  };
}

export default function katex(options: KatexOptions = {}): MdastPluginDefinition {
  const throwOnError = options.throwOnError ?? false;

  return defineMdastPlugin({
    name: "katex",
    math(node) {
      return toKatexComponent(node.value, true, throwOnError);
    },
    inlineMath(node) {
      return toKatexComponent(node.value, false, throwOnError);
    },
  });
}
