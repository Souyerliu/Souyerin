// cannot use path alias here because unocss can not resolve it
import { defineConfig } from "./toolkit/themeConfig.ts";

export default defineConfig({
  siteName: "Souyer's Blog",
  locale: "zh-CN",
  nav: [
    { href: "/", text: "首页", icon: "i-ri-home-line" },
    {
      text: "文章",
      href: "/random/",
      icon: "i-ri-quill-pen-fill",
      dropbox: {
        enable: true,
        items: [
          { href: "/categories/", text: "分类", icon: "i-ri-book-shelf-fill" },
          { href: "/tags/", text: "标签", icon: "i-ri-price-tag-3-fill" },
          { href: "/archives/", text: "归档", icon: "i-ri-archive-line" },
        ],
      },
    },
    { text: "友链", href: "/friends/", icon: "i-ri-link" },
    { text: "动态", href: "/moments/", icon: "i-ri-chat-quote-line" },
    { text: "关于", href: "/about/", icon: "i-ri-user-line" },
    {
      text: "副站",
      href: "https://souyerin.netlify.app",
      icon: "i-ri-external-link-line",
    },
  ],
  brand: {
    title: "Souyer's Blog",
    subtitle: "Souyer的博客",
    logo: "",
  },
  cover: {
    enable: true,
    preload: true,
    advancedCarousel: false,
    fixedCover: { enable: false },
    nextGradientCover: false,
  },
  sidebar: {
    author: "Souyer",
    description: "日日进化中！",
    social: {
      github: { url: "https://github.com/Souyerliu", icon: "i-ri-github-fill" },
      bilibili: {
        url: "https://space.bilibili.com/474896469",
        icon: "i-ri-bilibili-fill",
      },
      netease: {
        url: "https://music.163.com/#/user/home?id=1478960573",
        icon: "i-ri-netease-cloud-music-line",
      },
      email: { url: "mailto:zsy142857@126.com", icon: "i-ri-mail-line" },
    },
  },
  footer: {
    since: 2025,
    icon: { name: "sakura rotate", color: "var(--color-pink)" },
    count: true,
    powered: true,
    icp: {
      enable: true,
      icpnumber: "萌ICP备20260616号",
      icpurl: "https://icp.gov.moe/?keyword=20260616",
    },
  },
  tagCloud: {
    startColor: "var(--grey-6)",
    endColor: "var(--color-blue)",
  },
  widgets: {
    randomPosts: true,
    recentComments: true,
    recentCommentsLimit: 10,
    recentCommentsSiteRole: "main",
  },
  comments: {
    enable: true,
    waline: {
      serverURL: "https://souyerincomments.dpdns.org",
      lang: "zh-CN",
      dark: 'html[data-theme="dark"]',
      emoji: [
        "https://fastly.jsdelivr.net/npm/@waline/emojis@1.1.0/weibo",
        "https://fastly.jsdelivr.net/npm/@waline/emojis@1.1.0/alus",
        "https://fastly.jsdelivr.net/npm/@waline/emojis@1.1.0/bilibili",
        "https://fastly.jsdelivr.net/npm/@waline/emojis@1.1.0/qq",
        "https://fastly.jsdelivr.net/npm/@waline/emojis@1.1.0/tieba",
        "https://fastly.jsdelivr.net/npm/@waline/emojis@1.1.0/tw-emoji",
      ],
    },
  },
  home: {
    selectedCategories: [
      { name: "自然语言处理", cover: "/images/NLP.webp" },
      { name: "CS61B", cover: "/images/61B-header.webp" },
      { name: "机器学习方法", cover: "/images/ml.webp" },
    ],
    pageSize: 10,
    // 首页及分页文章卡片优先显示本地数据库中的 AI 摘要
    excerptSource: "ai",
    title: { behavior: "default", customTitle: "" },
  },
  layout: {
    mode: "three-column",
    rightSidebar: {
      order: ["announcement", "search", "calendar", "recentMoments", "randomPosts", "tagCloud"],
      announcement: true,
      search: true,
      calendar: true,
      recentMoments: true,
      randomPosts: true,
      tagCloud: true,
    },
  },
  friends: {
    title: "友链",
    description: "Souyer 的朋友与常用工具。",
    comments: false,
    personal: [
      {
        url: "https://cosx.org/",
        title: "统计之都",
        desc: "一个旨在推广与应用统计学知识的网站和社区。",
        author: "COS",
        avatar: "https://cosx.org/img/logo.png",
        color: "#8C1F22",
      },
      {
        url: "https://sqzr2319.github.io/",
        title: "sqzr2319's Blog",
        author: "sqzr2319",
        desc: "愿群星于道路中寻见你。",
        avatar: "https://sqzr2319.github.io/img/sqzr2319.png",
        color: "#aadbfa",
      },
      {
        url: "https://cloudingyu.github.io/",
        title: "CloudingYu的博客",
        author: "CloudingYu",
        desc: "懒惰骄傲不耐烦",
        avatar: "https://cloudingyu.github.io/img/profile_pic.jpg",
        color: "#f4b677",
      },
      {
        url: "https://myecnu.org/",
        title: "ECNU·驿站",
        author: "zeyi",
        desc: "软院同学制作的 ECNU 学习资源集合。",
        avatar:
          "https://avatars.githubusercontent.com/u/229353891?s=400&u=402f153bc2eeeeb04db6bb419dd231173dd0b045&v=4",
        color: "#cc002c",
      },
    ],
    tools: [
      {
        url: "https://karnaughmapsolver.com/zh",
        title: "卡诺图求解器",
        desc: "不止求解卡诺图，数电好帮手。",
        author: "KMap-Solver",
        avatar: "https://karnaughmapsolver.com/favicon.ico",
        color: "#Fa9B57",
      },
      {
        url: "https://wavedrom.com/editor.html",
        title: "Wavedrom",
        author: "Aliaksei Chapyzhenka",
        desc: "基于 JavaScript 的时序电路波形图绘制工具。",
        avatar: "https://wavedrom.com/images/favicon.ico",
        color: "#141414",
      },
      {
        url: "https://souyerliu.github.io/slp3_translation/",
        title: "SLP3 中文翻译",
        author: "Souyer",
        desc: "笔者使用 AI 翻译的中文版 SLP3 教材。",
        avatar: "https://web.stanford.edu/favicon.ico",
        color: "#79e16f",
      },
    ],
  },
  copyright: {
    license: "CC-BY-NC-SA-4.0",
    show: true,
  },
});
