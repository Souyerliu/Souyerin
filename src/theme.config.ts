// cannot use path alias here because unocss can not resolve it
import { defineConfig } from "./toolkit/themeConfig";

export default defineConfig({
  siteName: "Souyer's Blog",
  locale: "zh-CN", // 网站语言: "zh-CN" | "en"
  nav: [
    {
      href: "/",
      text: "首页",
      icon: "i-ri-home-line",
    },
    {
      text: "文章",
      href: "/random/",
      icon: "i-ri-quill-pen-fill",
      dropbox: {
        enable: true,
        items: [
          {
            href: "/categories/",
            text: "分类",
            icon: "i-ri-book-shelf-fill",
          },
          {
            href: "/tags/",
            text: "标签",
            icon: "i-ri-price-tag-3-fill",
          },
          {
            href: "/archives/",
            text: "归档",
            icon: "i-ri-archive-line",
          },
        ],
      },
    },
    // {
    //   text: "项目",
    //   href: "/notebooks/",
    //   icon: "i-ri-booklet-line",
    // },
    {
      text: "友链",
      href: "/friends/",
      icon: "i-ri-link",
    },
    {
      text: "动态",
      href: "/moments/",
      icon: "i-ri-chat-quote-line",
    },
    {
      text: "关于",
      href: "/about/",
      icon: "i-ri-user-line",
    },
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
    // 固定封面模式（可选）：
    // - enable: 是否启用固定封面
    // - url: 推荐填 "cover-1" ~ "cover-6"（来自 src/components/Images.astro 预设），
    //        或者填 public 路径/远程 URL（会使用 <img> 兜底渲染）
    fixedCover: {
      enable: false,
      //url: "cover-4",
    },
    // gradient: true, // 渐变模式
    nextGradientCover: false, // 文章导航使用渐变背景
  },
  sidebar: {
    author: "Souyer",
    description: "日日进化中！",
    social: {
      github: {
        url: "https://github.com/Souyerliu",
        icon: "i-ri-github-fill",
      },
      bilibili: {
        url: "https://space.bilibili.com/474896469",
        icon: "i-ri-bilibili-fill",
      },
      netease: {
        url: "https://music.163.com/#/user/home?id=1478960573",
        icon: "i-ri-netease-cloud-music-line",
      },
      email: {
        url: "mailto:zsy142857@126.com",
        icon: "i-ri-mail-line",
      },
    },
  },
  footer: {
    since: 2025,
    icon: {
      name: "sakura rotate",
      color: "var(--color-pink)",
    },
    count: true,
    powered: true,
    icp: {
      enable: true,
      // icon: '/beian-icon.png',
      icpnumber: "萌ICP备20260616号",
      icpurl: "https://icp.gov.moe/?keyword=20260616",
      // beian: '网安备案号',
      // recordcode: 'xxxxx',
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
  },
  comments: {
    enable: true,
    waline: {
      // 替换为你的 Waline 服务端地址，例如: https://comments.example.com
      serverURL: "https://souyerincomments.dpdns.org",
      // 推荐与站点语言保持一致
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
  hyc: {
    // HYC 扩展总开关：关闭后其所有子功能不可用
    enable: true,
    aiSummary: {
      // AI 摘要卡片开关（受 hyc.enable 总开关控制）
      enable: true,
      // 卡片标题
      title: "AI 摘要",
      // 是否显示摘要使用的模型名称
      showModel: true,
    },
    aiRecommend: {
      // AI 相近文章推荐开关（受 hyc.enable 总开关控制）
      enable: false,
      // 默认展示前 3 篇
      limit: 3,
      // 最低相似度阈值（0.4 = 40%）
      minSimilarity: 0.4,
    },
  },
  nyxPlayer: {
    enable: true,
    preset: "shokax",
    darkModeTarget: ':root[data-theme="dark"]',
    urls: [
      {
        name: "Souyer的歌单",
        url: "https://music.163.com/#/playlist?id=2257046115",
      },
    ],
  },
  visibilityTitle: {
    enable: true,
    leaveTitle: "哦内盖~",
    returnTitle: "祝你幸福。",
    restoreDelay: 3000,
  },
  home: {
    selectedCategories: [
      {
        name: "自然语言处理",
        cover: "/images/NLP.webp",
      },
      {
        name: "CS61B",
        cover: "/images/61B-header.webp",
      },
      {
        name: "CS127",
        cover: "/images/EECS127.webp",
      },
      {
        name: "机器学习方法",
        cover: "/images/ml.webp",
      },
    ],
    pageSize: 10,
    title: {
      behavior: "default",
      customTitle: "",
    },
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
    description: "卡片式展示，支持站点预览与主题色点缀。",
    // avatar: "https://example.com/your-avatar.png",
    // color: "var(--color-pink)",
    // siteImage: "https://example.com/your-site-preview.png",
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
        desc: "软院同学制作的ECNU学习资源集合。",
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
        desc: "基于javascript的时序电路波形图绘制工具，数电好帮手×2。",
        avatar: "https://wavedrom.com/images/favicon.ico",
        color: "#141414",
      },
    ],
  },
  copyright: {
    license: "CC-BY-NC-SA-4.0",
    show: true,
  },
  live2d: {
    // 是否启用 Live2D 看板娘
    enable: true,
    // 模型资源路径（本地 live2d-models 目录）
    cdnPath: "/live2d-models/",
    // 默认显示第几个模型（1 = Soyo）
    modelId: 1,
    // 是否允许拖动
    drag: true,
    // 关闭后是否显示重新唤起按钮
    showToggleAfterQuit: true,
    // 工具栏按钮（switch-model 切换角色，switch-texture 切换服装）
    tools: ["hitokoto", "asteroids", "switch-model", "switch-texture", "photo", "info", "quit"],
    // waifu-tips.json 路径（鼠标悬停/点击提示文案）
    waifuTipsPath: "/live2d-models/waifu-tips.json",
  },
});
