import { expect, test } from "@playwright/test";

test("@regression CS61B 第七章显示当前文章的 AI 摘要", async ({ page }) => {
  await page.goto("/posts/computer-science/cs61b/cs61b-chapter-7/");
  const summary = page.locator(".ai-summary-card__content");
  await expect(summary).toContainText("Java面向对象");
  await expect(summary).not.toContainText("人工智能技术");
});

test("@regression 多元统计分析文末标题、列表与公式正常渲染", async ({ page }) => {
  await page.goto("/posts/mathematics/多元统计分析-cheat-sheet/");
  const article = page.locator(".md");
  await expect(article.locator(".katex-error")).toHaveCount(0);
  await expect(article.getByRole("heading", { name: "矩阵不等式" })).toBeVisible();
  await expect(article.getByRole("heading", { name: "分块矩阵" })).toBeVisible();
  await expect(article.getByRole("heading", { name: "随机向量与矩阵" })).toBeVisible();
  const covariance = article
    .locator("li")
    .filter({ hasText: /^协方差矩阵：/ })
    .last();
  await expect(covariance).toBeVisible();
  await expect(covariance.locator(".katex")).toHaveCount(1);
  await expect(article).not.toContainText("$$");
});
