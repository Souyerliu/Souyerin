import { definePlugin, type PluginManifest } from "@hyacine/plugin-core";

export interface AiSummaryDirectOptions {
  /** 是否启用本地摘要数据库读取。 */
  enable?: boolean;
  /** 摘要卡片标题。 */
  title?: string;
  /** 是否显示摘要使用的模型名称。 */
  showModel?: boolean;
}

/**
 * 从本地摘要数据库读取文章摘要。
 *
 * 摘要由 scripts/generate-ai-summaries.mjs 预先生成，页面渲染阶段不发起网络请求。
 */
export default function aiSummaryDirect(options: AiSummaryDirectOptions = {}): PluginManifest {
  return definePlugin({
    name: "ai-summary-direct",
    version: "1.0.0",
    minRenderCapability: "ssr",
    supportedPlatforms: ["astro"],
    entry: [
      {
        name: "ai-summary-direct-ssr",
        type: "ssr",
        platform: "astro",
        injectPoint: "post-summary",
        requiresArticle: true,
        path: new URL("./AiSummaryCard.astro", import.meta.url).href,
        props: {
          enable: options.enable === true,
          title: options.title ?? "AI 摘要",
          showModel: options.showModel === true,
        },
      },
    ],
  });
}
