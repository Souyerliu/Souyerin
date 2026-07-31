<script lang="ts">
  import { init, type WalineInstance } from "@waline/client";
  import "@waline/client/style";
  import { onMount } from "svelte";
  import { t } from "@/i18n";

  interface Props {
    serverURL?: string;
    lang?: string;
    dark?: boolean | string;
    path?: string;
    pagePath?: string;
    emoji?: any;
  }

  const {
    serverURL = "",
    lang = "zh-CN",
    dark = false,
    path = "",
    pagePath = "",
    emoji = [],
  }: Props = $props();

  let walineEl = $state<HTMLDivElement | null>(null);

  /**
   * 根据 dark 配置值与当前 DOM 状态解析暗色模式是否启用。
   * - boolean: 直接返回
   * - "auto": 跟随系统 prefers-color-scheme
   * - CSS 选择器字符串: 检查选择器是否匹配（如 'html[data-theme="dark"]'）
   */
  function resolveDark(): boolean {
    if (typeof dark === "boolean") return dark;
    if (typeof dark === "string") {
      if (dark === "auto") {
        return window.matchMedia("(prefers-color-scheme: dark)").matches;
      }
      return !!document.querySelector(dark);
    }
    return false;
  }

  onMount(() => {
    if (!serverURL || !walineEl) return;

    const finalPath =
      path ||
      pagePath ||
      (typeof window !== "undefined" ? window.location.pathname : "/");

    const waline: WalineInstance | null = init({
      el: walineEl,
      serverURL,
      path: finalPath,
      lang,
      dark: resolveDark(),
      emoji,
    });

    // 监听 <html data-theme> 变化，同步更新 Waline 暗色模式
    const observer = new MutationObserver(() => {
      waline?.update?.({ dark: resolveDark() });
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    return () => {
      observer.disconnect();
      waline?.destroy();
    };
  });
</script>

{#if serverURL}
  <div bind:this={walineEl}></div>
{:else}
  <div class="waline-disabled">{t("footer.walineNotConfigured")}</div>
{/if}

<style>
  .waline-disabled {
    border: 1px dashed var(--grey-4);
    color: var(--grey-5);
    border-radius: 0.5rem;
    padding: 1rem;
    text-align: center;
    font-size: 0.875rem;
  }
</style>
