import type { PluginManifest } from "@hyacine/core";

export interface SiteUptimeOptions {
  /** 建站时间，需为合法的 Date 字符串。 */
  siteCreatedAt: string;
  /** 显示在运行时间之前的文案。 */
  prefixText?: string;
}

export default (options: SiteUptimeOptions): PluginManifest => {
  const createdDate = new Date(options.siteCreatedAt);
  if (Number.isNaN(createdDate.getTime())) {
    throw new Error(
      `[site-uptime] Invalid siteCreatedAt: "${options.siteCreatedAt}". Please provide a valid date string.`,
    );
  }

  return {
    name: "site-uptime-local",
    version: "1.0.0",
    minRenderCapability: "runtime-only",
    entry: [
      {
        type: "runtime-only",
        injectPoint: "footer-status",
        path: new URL("./runtime.ts", import.meta.url).href,
        name: "site-uptime-runtime",
        options: {
          siteCreatedAt: options.siteCreatedAt,
          prefixText: options.prefixText || "该站点已经存在了",
        },
      },
    ],
  };
};
