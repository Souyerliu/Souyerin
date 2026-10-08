import { expect, test } from "@playwright/test";
import tips from "../../../public/live2d-models/waifu-tips.json" with { type: "json" };

/* oxlint-disable no-await-in-loop -- 必须顺序验证同一实例的角色切换。 */
test("@regression 三位角色切换后更新悬停、身体互动、工具和缓存的闲聊台词", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.clock.install();
  await page.addInitScript(() => {
    localStorage.setItem("modelId", "0");
    localStorage.setItem("modelTexturesId", "0");
    Math.random = () => 0;
  });
  // 保留真实 waifu-tips.js 的配置、事件、计时器和切换流程，只替代绘图核心。
  await page.route("**/live2d.min.js", (route) =>
    route.fulfill({ body: "", contentType: "text/javascript" }),
  );
  await page.route("**/live2dcubismcore.min.js", (route) =>
    route.fulfill({ body: "", contentType: "text/javascript" }),
  );
  await page.route("**/chunk/index.js", (route) =>
    route.fulfill({
      contentType: "text/javascript",
      body: "export default class { constructor(){this.gl=true;} async init(){} async changeModelWithJSON(){} destroy(){} mouseEvent(){} }",
    }),
  );
  await page.route("**/chunk/index2.js", (route) =>
    route.fulfill({
      contentType: "text/javascript",
      body: `export class AppDelegate {
      constructor(){this.subdelegates={at:()=>this.delegate};}
      initialize(){this.initializeSubdelegates();}
      initializeSubdelegates(){this.delegate={update(){},getCanvas:()=>document.getElementById('live2d'),getLive2DManager:()=>({onDrag(){}}),_view:{transformViewX:x=>x,transformViewY:y=>y}};}
      onMouseMove(){} onMouseEnd(){} transformOffset(){return {x:0,y:0};}
      run(){} changeModel(){} release(){}
    }`,
    }),
  );
  await page.goto("/about/");
  await expect(page.locator("#waifu-tool-switch-model")).toBeAttached();
  await page.evaluate(() => {
    const link = document.createElement("a");
    link.href = "/about/";
    link.id = "role-about-link";
    link.textContent = "关于";
    document.body.append(link);
  });
  const bubble = page.locator("#waifu-tips");
  for (const id of ["0", "1", "2", "0"] as const) {
    if ((await page.evaluate(() => localStorage.getItem("modelId"))) !== id) {
      await page.locator("#waifu-tool-switch-model").dispatchEvent("click");
      await expect.poll(() => page.evaluate(() => localStorage.getItem("modelId"))).toBe(id);
      await page.clock.runFor(1000);
      await expect(bubble).toHaveText(tips.models[Number(id)].message);
    }
    const clear = () => page.evaluate(() => sessionStorage.removeItem("waifu-message-priority"));
    await clear();
    // 先经过别的选择器，解除库按最后一次悬停选择器去重的缓存。
    await page.locator("#waifu-tool-info").dispatchEvent("mouseover");
    await page.locator("#role-about-link").dispatchEvent("mouseover");
    await expect(bubble).toHaveText(
      tips.characterTips[id].mouseover.find((rule) => rule.selector === "a[href='/about/']")!
        .text[0],
    );
    for (const [event, key] of [
      ["live2d:hoverbody", "hoverBody"],
      ["live2d:tapbody", "tapBody"],
    ] as const) {
      await clear();
      await page.evaluate((name) => window.dispatchEvent(new Event(name)), event);
      await expect(bubble).toHaveText(tips.characterTips[id].message[key][0]);
    }
    await clear();
    await page.locator("#waifu-tool-switch-texture").dispatchEvent("click");
    await expect(bubble).toHaveText(tips.characterTips[id].message.changeSuccess);
    await clear();
    await page.locator("#waifu-tool-photo").dispatchEvent("click");
    await expect(bubble).toHaveText(tips.characterTips[id].message.photo);
    await clear();
    await page.clock.runFor(25_000);
    await expect(bubble).toHaveText(tips.characterTips[id].message.default[0]);
  }
  await page.evaluate(() => history.pushState({}, "", "/moments/"));
  await page.clock.runFor(1000);
  await expect(bubble).toHaveText(tips.characterTips["0"].pageWelcome.moments);
});
