import { expect, test } from "@playwright/test";

test("@regression Tabs 内折叠答案的 math 围栏正常渲染", async ({ page }) => {
  const response = await page.goto("/posts/test/");
  expect(response?.ok()).toBeTruthy();

  const tabs = page
    .locator(".md .tabs")
    .filter({ has: page.getByRole("tab", { name: "公式证明" }) });
  await expect(tabs.locator("code.language-math")).toHaveCount(0);
  await expect(tabs.locator(".katex-error")).toHaveCount(0);
  await expect(tabs.locator(".katex-display")).toHaveCount(2);

  await tabs.locator(".tab-item.active summary").click();
  await expect(tabs.locator(".tab-item.active .katex-display")).toBeVisible();
  await expect(tabs.locator(".tab-item.active .katex-html")).toBeVisible();

  await tabs.getByRole("tab", { name: "公式补充" }).click();
  await tabs.locator(".tab-item.active summary").click();
  await expect(tabs.locator(".tab-item.active .katex-display")).toBeVisible();
  await expect(tabs.locator(".tab-item.active .katex-html")).toBeVisible();
});
