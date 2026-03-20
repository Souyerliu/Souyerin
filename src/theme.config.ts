// cannot use path alias here because unocss can not resolve it
import { defineConfig } from "./toolkit/themeConfig";

export default defineConfig({
  siteName: "ShokaX",
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
      icpnumber: "津ICP备2022001375号",
      icpurl: "https://beian.miit.gov.cn/",
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
    enable: false,
    waline: {
      // 替换为你的 Waline 服务端地址，例如: https://comments.example.com
      serverURL: "",
      // 推荐与站点语言保持一致
      lang: "zh-CN",
    },
  },
  hyc: {
    // HYC 扩展总开关：关闭后其所有子功能不可用
    enable: false,
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
      enable: true,
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
        name: "CS61A",
        cover: "/images/61A-header.png",
      }, 
      { 
        name: "人工智能导论" ,
        cover: "/images/cover.jpg",
      },
      {
        name: "CS70",
        cover: "/images/penguin_and_pigeon.png",
      },
      {
        name: "CS127",
        cover: "/images/EECS127.png",
      }
    ],
    pageSize: 5,
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
    links: [
      {
        url: "https://astro.build/",
        title: "Astro",
        desc: "全站体验轻快的静态站点框架，适合内容型站点与博客。",
        author: "Astro Team",
        avatar: "https://avatars.githubusercontent.com/u/44914786?s=200&v=4",
        color: "var(--color-orange)",
        siteImage: "https://astro.build/assets/press/astro-logo-dark.svg",
      },
      {
        url: "https://svelte.dev/",
        title: "Svelte",
        desc: "编译时框架，现代与简洁，组件写起来很顺手。",
        author: "Svelte Team",
        avatar: "https://avatars.githubusercontent.com/u/23617963?s=200&v=4",
        color: "var(--color-red)",
      },
      {
        url: "https://vite.dev/",
        title: "Vite",
        desc: "快速的前端开发构建工具，HMR 体验很棒。",
        author: "Vite Team",
        avatar: "https://avatars.githubusercontent.com/u/65625612?s=200&v=4",
        color: "var(--color-blue)",
      },
      {
        url: "https://bun.sh/",
        title: "Bun",
        desc: "一体化 JavaScript 运行时，速度与工具链兼备。",
        author: "Bun Team",
        avatar: "https://avatars.githubusercontent.com/u/108928776?s=200&v=4",
        color: "var(--color-green)",
        siteImage: "https://bun.sh/logo.svg",
      },
      {
        url: "https://cosx.org/",
        title: "统计之都",
        desc: "一个旨在推广与应用统计学知识的网站和社区。",
        author: "COS",
        avatar: "https://cosx.org/img/logo.png",
        color: "#8C1F22",
      },
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
      {
        url: "https://myecnu.org/",
        title: "ECNU·驿站",
        author: "zeyi",
        desc: "软院同学制作的ECNU学习资源集合。",
        avatar: "https://avatars.githubusercontent.com/u/229353891?s=400&u=402f153bc2eeeeb04db6bb419dd231173dd0b045&v=4",
        color: "#cc002c",
      },
    ],
  },
  copyright: {
    license: "CC-BY-NC-SA-4.0",
    show: true,
  },
});
