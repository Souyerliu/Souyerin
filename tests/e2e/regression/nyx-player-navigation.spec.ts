import { expect, test } from "@playwright/test";

test("@regression 跨页导航后音乐播放器仍可反复显示和隐藏", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.route("https://meting.api.zkz098.cn/v1/playlists/**", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        code: 0,
        data: {
          songs: [
            {
              id: "1",
              name: "测试歌曲",
              artist: ["测试歌手"],
              pic_url: "",
            },
          ],
        },
      }),
    });
  });
  await page.route("https://meting.api.zkz098.cn/v1/songs/**/lyric**", async (route) => {
    await route.fulfill({ contentType: "text/plain", body: "[00:00.00]" });
  });

  await page.goto("/");

  const showButton = page.getByRole("button", { name: "显示或隐藏播放器" });
  await showButton.click();

  const player = page.locator(".nyx-player");
  await expect(player).toBeVisible();
  await expect(player.locator("..").locator("..")).toHaveAttribute("id", "nyx-player-host");

  await page.locator('a[href="/about/"]').first().click();
  await expect(page).toHaveURL(/\/about\/$/);
  await expect(player).toBeVisible();

  await showButton.click();
  await expect(player).toHaveCount(0);

  await showButton.click();
  await expect(player).toBeVisible();
  expect(pageErrors).not.toContainEqual(expect.stringContaining("insertBefore"));
});
