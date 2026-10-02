import { expect, test } from "@playwright/test";
import sharp from "sharp";

/* oxlint-disable no-await-in-loop -- 模型切换必须按用户操作顺序执行，无法并行。 */

test("@regression Live2D 跨版本切换后遮罩正常且画布尺寸保持一致", async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.addInitScript(() => {
    localStorage.setItem("modelId", "1");
    localStorage.setItem("modelTexturesId", "0");
  });
  await page.goto("/");
  const canvas = page.locator("#live2d");

  const expectRendered = async () => {
    await expect(canvas).toBeVisible();
    await expect
      .poll(
        async () => {
          const data = await canvas.evaluate((element) => {
            if (!(element instanceof HTMLCanvasElement)) throw new Error("缺少 Live2D 画布");
            return element.toDataURL().split(",")[1];
          });
          const { data: pixels } = await sharp(Buffer.from(data, "base64"))
            .ensureAlpha()
            .raw()
            .toBuffer({ resolveWithObject: true });
          let visiblePixels = 0;
          for (let index = 3; index < pixels.length; index += 4) {
            if (pixels[index] > 0) visiblePixels++;
          }
          return visiblePixels;
        },
        { timeout: 30_000 },
      )
      .toBeGreaterThan(1000);
    await expect(canvas).toHaveCSS("width", "300px");
    await expect(canvas).toHaveCSS("height", "300px");
    // 等待完整一帧，检查切回旧核心时是否仍在引用另一画布的遮罩。
    const error = await canvas.evaluate(async (element) => {
      if (!(element instanceof HTMLCanvasElement)) throw new Error("缺少 Live2D 画布");
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const gl = element.getContext("webgl2");
      if (!gl) throw new Error("WebGL 2 不可用");
      return gl.getError();
    });
    expect(error).toBe(0);
  };

  await expectRendered();
  let gazeChecked = false;
  // 两次跨版本往返，覆盖 Cubism 2 → 3 → 2 及重新使用旧核心的情形。
  for (const id of [2, 0, 1, 2, 0, 1]) {
    const previousCanvas = await canvas.elementHandle();
    await page.locator("#waifu-tool-switch-model").click();
    await expect.poll(() => page.evaluate(() => localStorage.getItem("modelId"))).toBe(String(id));
    if (id !== 1) {
      await expect
        .poll(() => previousCanvas?.evaluate((element) => element.isConnected))
        .toBe(false);
    }
    await expectRendered();
    if (id === 2 && !gazeChecked) {
      gazeChecked = true;
      const runtime = await page
        .locator('script[src$="/waifu-tips.js"]')
        .first()
        .getAttribute("src");
      if (!runtime) throw new Error("缺少 Live2D 运行时地址");
      await page.evaluate(async (url) => {
        const { AppDelegate } = await import(url.replace("/waifu-tips.js", "/chunk/index2.js"));
        const original = AppDelegate.prototype.transformOffset;
        AppDelegate.prototype.transformOffset = function (event: MouseEvent) {
          const offset = original.call(this, event);
          document.documentElement.dataset.live2dPointer = JSON.stringify(offset);
          return offset;
        };
        document.body.style.minHeight = "6000px";
        const widget = document.getElementById("waifu");
        if (widget) widget.style.transition = "none";
      }, runtime);
      await page.mouse.move(700, 150);
      const beforeScroll = await page.locator("html").getAttribute("data-live2d-pointer");
      if (!beforeScroll) throw new Error("模型尚未接收光标位置");
      await page.evaluate(() => window.scrollTo(0, 2500));
      await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(2500);
      await page.mouse.move(701, 150);
      await page.mouse.move(700, 150);
      await expect(page.locator("html")).toHaveAttribute("data-live2d-pointer", beforeScroll);
      await page.evaluate(() => {
        window.scrollTo(0, 0);
        document.body.style.minHeight = "";
      });
    }
  }
});
