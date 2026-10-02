import { describe, expect, it } from "vitest";
import tips from "../../public/live2d-models/waifu-tips.json";
import { resolveLive2DWelcome } from "./live2d-welcome";

describe.each(["0", "1", "2"] as const)("角色 %s 的页面欢迎语", (id) => {
  it.each([
    "/",
    "/page/2/",
    "/about/",
    "/moments/",
    "/moments/2/",
    "/friends/",
    "/archives/",
    "/archives/2026/10/",
    "/tags/",
    "/tags/学习/",
    "/categories/",
    "/categories/计算机/",
    "/statistics/",
    "/random/",
    "/unknown/",
  ])("%s 使用栏目台词而非阅读模板", (path) => {
    const text = resolveLive2DWelcome(tips, id, path);
    expect(text).not.toContain("$1");
    expect(text).not.toContain("欢迎阅读");
    expect(text).not.toContain("这篇");
    expect(text.length).toBeGreaterThan(0);
  });

  it("普通文章和名为 about 的文章均保留角色阅读模板", () => {
    for (const path of ["/posts/deep-learning/chapter-17/", "/posts/about/"]) {
      expect(resolveLive2DWelcome(tips, id, path)).toBe(tips.characterTips[id].message.welcome);
    }
  });
});

it("未知角色与缺失配置使用通用栏目台词，未知页面不误判为文章", () => {
  expect(resolveLive2DWelcome(tips, "99", "/about/")).toBe(tips.pageWelcome.about);
  expect(resolveLive2DWelcome({}, "99", "/unknown/")).not.toContain("阅读");
});
