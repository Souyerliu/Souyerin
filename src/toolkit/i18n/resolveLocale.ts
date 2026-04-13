export const SUPPORTED_LOCALES = ["zh-CN", "en"] as const;

export type ResolvedLocale = (typeof SUPPORTED_LOCALES)[number];

const SUPPORTED_LOCALE_SET = new Set<string>(SUPPORTED_LOCALES);

export function resolveLocale(locale: unknown, fallback: ResolvedLocale = "zh-CN"): ResolvedLocale {
  if (typeof locale === "string" && SUPPORTED_LOCALE_SET.has(locale)) {
    return locale as ResolvedLocale;
  }

  return fallback;
}
