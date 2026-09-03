import { onCleanup, onMount } from "solid-js";
import {
  init,
  type WalineInitOptions,
  type WalineInstance,
} from "@waline/client";
import "@waline/client/style";

import { t } from "@/i18n";

interface WalineCommentsProps {
  serverURL?: string;
  lang?: string;
  dark?: boolean | string;
  path?: string;
  pagePath?: string;
  emoji?: WalineInitOptions["emoji"];
}

function resolveDarkMode(dark: boolean | string | undefined): boolean {
  if (typeof dark === "boolean") return dark;

  if (typeof dark === "string") {
    if (dark === "auto") {
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    return Boolean(document.querySelector(dark));
  }

  return false;
}

export default function WalineComments(props: WalineCommentsProps) {
  let walineElement: HTMLDivElement | undefined;

  onMount(() => {
    if (!props.serverURL || !walineElement) return;

    const finalPath =
      props.path ||
      props.pagePath ||
      (typeof window !== "undefined" ? window.location.pathname : "/");

    const waline: WalineInstance | null = init({
      el: walineElement,
      serverURL: props.serverURL,
      path: finalPath,
      lang: props.lang || "zh-CN",
      dark: resolveDarkMode(props.dark),
      emoji: props.emoji,
    });

    const observer = new MutationObserver(() => {
      waline?.update?.({
        dark: resolveDarkMode(props.dark),
        emoji: props.emoji,
      });
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    onCleanup(() => {
      observer.disconnect();
      waline?.destroy();
    });
  });

  return props.serverURL ? (
    <div ref={(element) => (walineElement = element)} />
  ) : (
    <div class="waline-disabled">{t("footer.walineNotConfigured")}</div>
  );
}
