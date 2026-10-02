import { expect, test } from "@playwright/test";

/* oxlint-disable no-await-in-loop -- 切页与入场必须按用户操作顺序执行。 */
test("@regression 由乃在切页入场期间收到光标后，静止光标仍对应最终画布位置", async ({ page }) => {
  test.setTimeout(90_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.addInitScript(() => {
    localStorage.setItem("modelId", "2");
    localStorage.setItem("modelTexturesId", "0");
  });
  await page.goto("/posts/live2d-gaze-first/");
  await expect(page.locator("#live2d")).toBeVisible();
  const runtimeUrl = await page.locator('script[src$="/waifu-tips.js"]').getAttribute("src");
  if (!runtimeUrl) throw new Error("缺少 Live2D 运行时");
  await page.evaluate(async (url) => {
    const { AppDelegate } = await import(url.replace("/waifu-tips.js", "/chunk/index2.js"));
    const transform = AppDelegate.prototype.transformOffset;
    AppDelegate.prototype.transformOffset = function (event: MouseEvent) {
      const result = transform.call(this, event);
      document.documentElement.dataset.gazeOffset = JSON.stringify(result);
      return result;
    };
  }, runtimeUrl);
  for (const slug of ["second", "third"]) {
    await page.evaluate((next) => {
      // ClientRouter 保留画布并搬移宿主；独立验证组件的切页监听与入场动画。
      const host = document.getElementById("live2d-persist-host");
      host?.remove();
      if (host) document.body.append(host);
      history.pushState({}, "", "/posts/live2d-gaze-" + next + "/");
      document.dispatchEvent(new Event("astro:after-swap"));
    }, slug);
    await expect
      .poll(() => page.locator("#waifu").evaluate((el) => el.getBoundingClientRect().top))
      .toBeGreaterThan(950);
    await page.mouse.move(700, 150);
    // 光标停住，等待画布归位。旧实现会永久保留入场中途的视线坐标。
    await expect
      .poll(() => page.locator("#waifu").evaluate((el) => el.getBoundingClientRect().top))
      .toBeLessThan(650);
    await expect
      .poll(async () => {
        return page.evaluate(() => {
          const canvas = document.getElementById("live2d");
          if (!(canvas instanceof HTMLCanvasElement)) return false;
          const actual = document.documentElement.dataset.gazeOffset;
          if (!actual) return false;
          const { y } = JSON.parse(actual);
          const rect = canvas.getBoundingClientRect();
          // Cubism 默认视图：画布 y 转换为 [-1, 1]，向上为正。
          const expectedY = 1 - (2 * (150 - rect.top)) / rect.height;
          return Math.abs(y - expectedY) < 0.01;
        });
      })
      .toBe(true);
  }
});
