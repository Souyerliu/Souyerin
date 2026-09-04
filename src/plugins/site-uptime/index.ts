import { definePlugin, type PluginManifest } from "@hyacine/plugin-core";

export interface SiteUptimeOptions {
  /** 建站时间，需为合法的 ISO 日期字符串。 */
  siteCreatedAt: string;
  /** 显示在运行时间之前的文案。 */
  prefixText?: string;
}

export default function siteUptime(options: SiteUptimeOptions): PluginManifest {
  const createdDate = new Date(options.siteCreatedAt);
  if (Number.isNaN(createdDate.getTime())) {
    throw new Error(`[site-uptime] Invalid siteCreatedAt: "${options.siteCreatedAt}"`);
  }

  return definePlugin({
    name: "site-uptime-local",
    version: "1.0.0",
    minRenderCapability: "runtime-only",
    entry: [
      {
        name: "site-uptime-runtime",
        type: "runtime-only",
        injectPoint: "footer-status",
        path: new URL("./runtime.ts", import.meta.url).href,
        options: {
          siteCreatedAt: options.siteCreatedAt,
          prefixText: options.prefixText ?? "本站点已经存在了",
        },
      },
    ],
  });
}
