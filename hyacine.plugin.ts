import { defineConfig } from "@hyacine/plugin-core";
import siteUptime from "./src/plugins/site-uptime";
import mouseFirework from "@hyacine/plugin-mouse-firework";
import articleAgeWarning from "@hyacine/plugin-article-age-warning";
import vercount from "@hyacine/plugin-vercount";
import analytics from "@hyacine/plugin-analytics";
import walineComments from "@hyacine/plugin-waline-comments";
import aiContent from "@hyacine/plugin-ai-content";
import aiSummaryDirect from "./src/plugins/ai-summary-direct";
import visibilityTitle from "@hyacine/plugin-visibility-title";
import nyxPlayer from "@hyacine/plugin-nyx-player";
import articleStatistics from "@hyacine/plugin-article-statistics";

export default defineConfig({
  injectPoints: {
    "footer-status": {
      selector: "#footer .status",
      position: "append",
    },
    "post-header": {
      selector: "article.post header",
      position: "after",
    },
    "post-footer": {
      selector: "article.post .body",
      position: "after",
    },
  },
  plugins: [
    siteUptime({
      siteCreatedAt: "2025-09-24T00:00:00Z",
      prefixText: "本站点已经存在了",
    }),
    mouseFirework({
      colors: [
        "rgba(255,182,185,.9)",
        "rgba(250,227,217,.9)",
        "rgba(187,222,214,.9)",
        "rgba(138,198,209,.9)",
      ],
      count: 30,
      radius: 16,
    }),
    articleAgeWarning({
      maxAgeDays: 180,
    }),
    vercount(),
    analytics({
      googleAnalytics: {
        measurementId: "",
      },
      umami: {
        websiteId: "",
        scriptUrl: "",
      },
    }),
    walineComments({
      serverURL: "https://souyerincomments.dpdns.org",
      lang: "zh-CN",
    }),
    aiContent({
      enable: false,
      aiSummary: {
        // 摘要改由本地插件读取旧版 OpenAI 兼容配置，避免重复渲染。
        enable: false,
        title: "AI 摘要",
        showModel: true,
      },
      aiRecommend: {
        enable: false,
        limit: 3,
        minSimilarity: 0.4,
      },
    }),
    aiSummaryDirect({
      enable: true,
      title: "AI 摘要",
      showModel: true,
    }),
    visibilityTitle({
      enable: true,
      leaveTitle: "哦内盖~",
      returnTitle: "祝你幸福。",
      restoreDelay: 3000,
    }),
    nyxPlayer({
      enable: true,
      urls: [
        {
          name: "Souyer的歌单",
          url: "https://music.163.com/#/playlist?id=2257046115",
        },
      ],
      preset: "shokax",
      darkModeTarget: ":root[data-theme=dark]",
      // 通过 Cloudflare Pages Function 同源代理，避免自定义域名触发第三方 CORS 限制。
      metingBaseURL: "/api/music",
      metingUrlSource: "outer",
    }),
    articleStatistics(),
  ],
});
