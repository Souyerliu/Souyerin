import { expect, test } from "@playwright/test";
import tips from "../../../public/live2d-models/waifu-tips.json" with { type: "json" };

// 欢迎语由本站控制。替代远程绘图核心，稳定验证首次加载、切页与角色差异。
const runtime = `window.initWidget = async (config) => {
  const tips = await (await fetch(config.waifuPath)).json();
  const widget = document.createElement("div");
  widget.id = "waifu";
  widget.className = "waifu-active";
  widget.innerHTML = '<div id="waifu-tips" class="waifu-tips-active"></div>';
  widget.firstElementChild.innerHTML = tips.message.welcome.split("$1").join(document.title);
  document.body.append(widget);
};`;

for (const id of ["0", "1", "2"] as const) {
  test(`@regression 角色 ${id} 区分直接打开栏目与站内切页欢迎语`, async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.addInitScript((modelId) => {
      localStorage.setItem("modelId", modelId);
      localStorage.setItem("modelTexturesId", "0");
    }, id);
    await page.route("**/waifu-tips.js", (route) =>
      route.fulfill({
        contentType: "text/javascript",
        body: runtime,
      }),
    );
    await page.route("**/chunk/index2.js", (route) =>
      route.fulfill({
        contentType: "text/javascript",
        body: "export class AppDelegate { onMouseEnd() {} }",
      }),
    );
    await page.goto("/about/");
    const bubble = page.locator("#waifu-tips");
    await expect(bubble).toHaveText(tips.characterTips[id].pageWelcome.about);
    // pushState 模拟 ClientRouter 保留宿主后的路径变化，直接测试本站的切页监听。
    await page.evaluate(() => {
      history.pushState({}, "", "/moments/2/");
      document.title = "动态";
    });
    await expect(bubble).toHaveText(tips.characterTips[id].pageWelcome.moments);
    await page.evaluate(() => {
      history.pushState({}, "", "/posts/hello-world/");
      document.title = "Hello World!";
    });
    const articleText = tips.characterTips[id].message.welcome
      .replace("$1", "Hello World!")
      .replace(/<[^>]*>/g, "");
    await expect(bubble).toHaveText(articleText);
    await page.evaluate(() => history.pushState({}, "", "/"));
    await expect(bubble).toHaveText(tips.characterTips[id].pageWelcome.home);
  });
}
