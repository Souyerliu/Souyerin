interface WelcomeTips {
  message?: { welcome?: string };
  pageWelcome?: Record<string, string>;
  characterTips?: Record<
    string,
    {
      message?: { welcome?: string };
      pageWelcome?: Record<string, string>;
    }
  >;
}

/** 文章才使用阅读欢迎语，其余路由按栏目选择角色台词。 */
export function resolveLive2DWelcome(
  tips: WelcomeTips,
  characterId: string,
  pathname: string,
): string {
  const segments = pathname.split(/[?#]/)[0].split("/").filter(Boolean);
  const character = tips.characterTips?.[characterId];
  if (segments[0] === "posts" && segments.length > 1) {
    return character?.message?.welcome || tips.message?.welcome || "欢迎阅读<span>「$1」</span>。";
  }
  const page = !segments.length || segments[0] === "page" ? "home" : segments[0];
  return (
    character?.pageWelcome?.[page] ||
    tips.pageWelcome?.[page] ||
    character?.pageWelcome?.default ||
    tips.pageWelcome?.default ||
    "欢迎来到这里，随意逛逛吧。"
  );
}
