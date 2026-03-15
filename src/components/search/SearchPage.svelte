<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { getT } from "@/i18n";
  import themeConfig from "@/theme.config";
  import pagefindCssHref from "@pagefind/default-ui/css/ui.css?url";

  const isDev = import.meta.env.DEV;
  const searchPanelTransitionMs = 350;

  interface Props {
    selector?: string | HTMLElement;
    showSearch?: boolean;
  }

  let { selector = undefined, showSearch = $bindable(false) }: Props = $props();
  const t = getT((themeConfig.locale as "zh-CN" | "en") || "zh-CN");

  let internalVisible = $state(false);
  let rendered = $state(false);
  let animatedVisible = $state(false);
  let cleanupListener: (() => void) | null = null;
  let cleanupKeyboard: (() => void) | null = null;
  let panelElement: HTMLDivElement | null = null;
  let hideTimeoutId: number | null = null;
  const visible = $derived(selector ? internalVisible : Boolean(showSearch));

  function clearHideTimeout() {
    if (hideTimeoutId === null || typeof window === "undefined") return;

    window.clearTimeout(hideTimeoutId);
    hideTimeoutId = null;
  }

  function openSearch() {
    if (selector) {
      internalVisible = true;
      return;
    }

    showSearch = true;
  }

  function closeSearch() {
    if (selector) {
      internalVisible = false;
      return;
    }

    showSearch = false;
  }

  function toggleVisibility() {
    if (selector) {
      internalVisible = !internalVisible;
      return;
    }

    showSearch = !showSearch;
  }

  function focusSearchInput() {
    if (typeof window === "undefined") return;

    window.requestAnimationFrame(() => {
      panelElement
        ?.querySelector<HTMLInputElement>(".pagefind-ui__search-input")
        ?.focus();
    });
  }

  function isEditableTarget(target: EventTarget | null) {
    if (!(target instanceof HTMLElement)) return false;

    return (
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      target.isContentEditable
    );
  }

  function loadPagefindCssAfterOnload() {
    return new Promise<void>((resolve) => {
      const existing = document.querySelector<HTMLLinkElement>(
        'link[data-pagefind-ui="true"]',
      );
      if (existing) {
        resolve();
        return;
      }

      const appendCss = () => {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = pagefindCssHref;
        link.dataset.pagefindUi = "true";
        link.onload = () => resolve();
        link.onerror = () => resolve();
        document.head.appendChild(link);
      };

      if (document.readyState === "complete") {
        appendCss();
        return;
      }

      window.addEventListener("load", appendCss, { once: true });
    });
  }

  async function initPagefind() {
    if (isDev) return;

    try {
      const [{ PagefindUI }] = await Promise.all([
        // @ts-expect-error no types for PagefindUI
        import("@pagefind/default-ui"),
        loadPagefindCssAfterOnload(),
      ]);
      new PagefindUI({ element: "#pagefind", showSubResults: true });
    } catch (error) {
      console.warn("Pagefind 初始化失败：", error);
    }
  }

  onMount(() => {
    initPagefind();

    // Setup selector listener
    if (selector) {
      let element: HTMLElement | null = null;

      if (typeof selector === "string") {
        element = document.querySelector(selector);
      } else if (selector instanceof HTMLElement) {
        element = selector;
      }

      if (element) {
        element.addEventListener("click", toggleVisibility);
        cleanupListener = () => {
          element?.removeEventListener("click", toggleVisibility);
        };
      } else {
        console.warn("Invalid selector provided for PagefindSearch component.");
      }
    }

    const handleKeydown = (event: KeyboardEvent) => {
      const isSearchShortcut =
        (event.ctrlKey || event.metaKey) &&
        event.key.toLowerCase() === "k" &&
        !event.altKey;

      if (isSearchShortcut && !isEditableTarget(event.target)) {
        event.preventDefault();
        openSearch();
        focusSearchInput();
        return;
      }

      if (event.key === "Escape") {
        closeSearch();
      }
    };

    window.addEventListener("keydown", handleKeydown);
    cleanupKeyboard = () => {
      window.removeEventListener("keydown", handleKeydown);
    };
  });

  onDestroy(() => {
    clearHideTimeout();

    if (cleanupListener) {
      cleanupListener();
      cleanupListener = null;
    }
    if (cleanupKeyboard) {
      cleanupKeyboard();
      cleanupKeyboard = null;
    }
  });

  $effect(() => {
    clearHideTimeout();

    if (visible) {
      rendered = true;

      if (typeof window === "undefined") {
        animatedVisible = true;
        return;
      }

      const frameId = window.requestAnimationFrame(() => {
        animatedVisible = true;
      });

      return () => {
        window.cancelAnimationFrame(frameId);
      };
    }

    animatedVisible = false;

    if (typeof window === "undefined") {
      rendered = false;
      return;
    }

    const currentTimeoutId = window.setTimeout(() => {
      rendered = false;
      hideTimeoutId = null;
    }, searchPanelTransitionMs);

    hideTimeoutId = currentTimeoutId;

    return () => {
      window.clearTimeout(currentTimeoutId);
      if (hideTimeoutId === currentTimeoutId) {
        hideTimeoutId = null;
      }
    };
  });

  $effect(() => {
    if (!visible) return;

    focusSearchInput();
  });

  $effect(() => {
    if (typeof document === "undefined" || !visible) return;

    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.body.style.overflow = originalOverflow;
    };
  });
</script>

<div
  class="search-shell"
  class:search-shell-visible={animatedVisible}
  hidden={!rendered}
  aria-hidden={!animatedVisible}
>
  <button
    type="button"
    class="search-overlay"
    aria-label="Close search overlay"
    onclick={closeSearch}
  ></button>

  <div
    bind:this={panelElement}
    class="pagefind-panel"
    role="dialog"
    aria-modal="true"
    aria-label="Search"
  >
    <div class="search-panel__ornament" aria-hidden="true">
      <span class="search-panel__icon i-ri-search-line"></span>
      <span class="search-panel__glow"></span>
    </div>

    <button
      type="button"
      class="search-panel__close"
      onclick={closeSearch}
      aria-label="Close search"
      aria-controls="pagefind"
    >
      <span class="i-ri-close-line"></span>
    </button>

    <div class="search-panel__body">
      {#if isDev}
        <div class="dev-tip">
          {t("search.devModeSkipped")}<br />
          {t("search.buildHint")}
        </div>
      {:else}
        <div id="pagefind"></div>
      {/if}
    </div>
  </div>
</div>

<style>
  .search-shell {
    position: fixed;
    inset: 0;
    z-index: var(--z-search-overlay);
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 4.5rem 1rem 1rem;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s ease;
  }

  .search-shell[hidden] {
    display: none !important;
  }

  .search-shell-visible {
    opacity: 1;
    pointer-events: auto;
  }

  .search-overlay {
    position: absolute;
    inset: 0;
    border: 0;
    padding: 0;
    cursor: pointer;
    background:
      radial-gradient(circle at top, var(--grey-1-a3), transparent 55%),
      var(--search-overlay-bg);
    backdrop-filter: blur(14px) saturate(140%);
    transition: opacity 0.3s ease;
  }

  .pagefind-panel {
    position: relative;
    z-index: var(--z-content);
    display: flex;
    min-height: min(32rem, calc(100vh - 6rem));
    max-height: calc(100vh - 6rem);
    width: min(100%, 72rem);
    flex-direction: column;
    overflow: hidden;
    border: 1px solid color-mix(in srgb, var(--grey-4) 20%, transparent);
    border-radius: 1.25rem;
    background: linear-gradient(
      145deg,
      color-mix(in srgb, var(--grey-1) 94%, var(--color-cyan-light) 6%),
      color-mix(in srgb, var(--grey-2) 92%, var(--color-pink-light) 8%)
    );
    box-shadow: var(--search-panel-shadow), var(--search-panel-inset);
    backdrop-filter: blur(20px) saturate(180%);
    transform: translateY(-1.5rem) scale(0.98);
    transition:
      transform 0.35s ease,
      box-shadow 0.35s ease;
  }

  .search-shell-visible .pagefind-panel {
    transform: translateY(0) scale(1);
    box-shadow:
      var(--search-panel-shadow-active), var(--search-panel-inset-active);
  }

  .search-panel__ornament {
    position: absolute;
    left: 1.5rem;
    top: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--color-red);
    pointer-events: none;
  }

  .search-panel__icon {
    display: inline-flex;
    font-size: 1.1rem;
    opacity: 0.95;
  }

  .search-panel__glow {
    height: 0.625rem;
    width: 4.5rem;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--color-red), transparent);
    opacity: 0.5;
  }

  .search-panel__close {
    position: absolute;
    right: 1rem;
    top: 1rem;
    z-index: var(--z-elevated);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    height: 2.75rem;
    width: 2.75rem;
    border: 1px solid color-mix(in srgb, var(--grey-4) 24%, transparent);
    border-radius: 999px;
    background: color-mix(in srgb, var(--grey-1) 82%, transparent);
    color: var(--text-color);
    cursor: pointer;
    transition:
      transform 0.2s ease,
      color 0.2s ease,
      border-color 0.2s ease,
      background 0.2s ease,
      box-shadow 0.2s ease;
  }

  .search-panel__close:hover,
  .search-panel__close:focus-visible {
    transform: translateY(-1px) scale(1.03);
    border-color: color-mix(in srgb, var(--color-red) 30%, var(--grey-4));
    background: color-mix(in srgb, var(--color-red-a1) 55%, var(--grey-1));
    color: var(--color-red);
    box-shadow: var(--search-accent-shadow);
    outline: none;
  }

  .search-panel__body {
    display: flex;
    flex: 1;
    min-height: 0;
    padding: 1rem;
    padding-top: 3.75rem;
  }

  .dev-tip {
    display: grid;
    flex: 1;
    place-items: center;
    border: 1px dashed color-mix(in srgb, var(--grey-4) 35%, transparent);
    border-radius: 1rem;
    background: linear-gradient(
      145deg,
      color-mix(in srgb, var(--grey-0) 84%, transparent),
      color-mix(in srgb, var(--grey-2) 90%, transparent)
    );
    color: var(--text-color);
    line-height: 1.9;
    padding: 2rem;
    text-align: center;
  }

  .pagefind-panel :global(#pagefind),
  .pagefind-panel :global(.pagefind-ui) {
    display: flex;
    flex: 1;
    min-height: 0;
    flex-direction: column;
    width: 100%;
  }

  .pagefind-panel :global(.pagefind-ui) {
    --pagefind-ui-scale: 1;
    --pagefind-ui-primary: var(--color-red);
    --pagefind-ui-text: var(--text-color);
    --pagefind-ui-background: transparent;
    --pagefind-ui-border: color-mix(in srgb, var(--grey-4) 25%, transparent);
    --pagefind-ui-tag: color-mix(in srgb, var(--grey-2) 90%, transparent);
    --pagefind-ui-border-width: 1px;
    --pagefind-ui-border-radius: 0.9rem;
    --pagefind-ui-font: inherit;
    overflow: hidden;
  }

  .pagefind-panel :global(.pagefind-ui__form) {
    position: relative;
  }

  .pagefind-panel :global(.pagefind-ui__form::before) {
    content: none;
    display: none;
    background: transparent;
  }

  .pagefind-panel :global(.pagefind-ui__search-input) {
    min-height: 3.5rem;
    border: 1px solid color-mix(in srgb, var(--grey-4) 24%, transparent);
    border-radius: 1rem;
    background: linear-gradient(
      145deg,
      color-mix(in srgb, var(--grey-0) 88%, transparent),
      color-mix(in srgb, var(--grey-2) 94%, transparent)
    );
    box-shadow: var(--search-input-inset), var(--search-input-shadow);
    color: var(--text-color);
    padding-inline: 1rem 3rem;
    transition:
      border-color 0.2s ease,
      box-shadow 0.2s ease,
      transform 0.2s ease;
  }

  .pagefind-panel :global(.pagefind-ui__search-input:focus) {
    border-color: color-mix(in srgb, var(--color-red) 38%, var(--grey-4));
    box-shadow:
      0 0 0 0.25rem color-mix(in srgb, var(--color-red-a1) 78%, transparent),
      var(--search-accent-shadow-soft);
    outline: none;
    transform: translateY(-1px);
  }

  .pagefind-panel :global(.pagefind-ui__search-input::placeholder) {
    color: var(--grey-5);
  }

  .pagefind-panel :global(.pagefind-ui__search-clear) {
    width: 2.75rem;
    color: var(--grey-5);
  }

  .pagefind-panel :global(.pagefind-ui__drawer) {
    display: block;
    flex: 1;
    min-height: 0;
    margin-top: 1rem;
    padding: 0;
    border: 0;
    background: transparent;
    overflow: auto;
    overscroll-behavior: contain;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: thin;
    scrollbar-color: color-mix(in srgb, var(--grey-4) 80%, transparent)
      transparent;
  }

  .pagefind-panel :global(.pagefind-ui__results-area) {
    min-height: 0;
    overflow: scroll;
    max-height: 40rem;
    scrollbar-width: thin;
  }

  .pagefind-panel :global(.pagefind-ui__results) {
    display: grid;
    gap: 0.9rem;
  }

  .pagefind-panel :global(.pagefind-ui__result) {
    position: relative;
    overflow: hidden;
    border: 1px solid color-mix(in srgb, var(--grey-4) 18%, transparent);
    border-radius: 1rem;
    background: linear-gradient(
      145deg,
      color-mix(in srgb, var(--grey-0) 88%, transparent),
      color-mix(in srgb, var(--grey-2) 94%, transparent)
    );
    box-shadow: var(--search-result-shadow);
    padding: 1rem 1.1rem;
    transition:
      transform 0.25s ease,
      box-shadow 0.25s ease,
      border-color 0.25s ease;
  }

  .pagefind-panel :global(.pagefind-ui__result::before) {
    content: "";
    position: absolute;
    inset: 0 auto auto 0;
    height: 3px;
    width: 100%;
    background: linear-gradient(
      90deg,
      color-mix(in srgb, var(--color-red) 70%, transparent),
      color-mix(in srgb, var(--color-blue) 55%, transparent),
      transparent
    );
    opacity: 0.7;
  }

  .pagefind-panel :global(.pagefind-ui__result:hover) {
    transform: translateY(-2px);
    border-color: color-mix(in srgb, var(--color-red) 24%, var(--grey-4));
    box-shadow: var(--search-result-shadow-hover);
  }

  .pagefind-panel :global(.pagefind-ui__result-link) {
    color: var(--text-color);
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.5;
    text-decoration: none;
    transition: color 0.2s ease;
  }

  .pagefind-panel :global(.pagefind-ui__result-link:hover) {
    color: var(--color-red);
  }

  .pagefind-panel :global(.pagefind-ui__result-excerpt),
  .pagefind-panel :global(.pagefind-ui__message),
  .pagefind-panel :global(.pagefind-ui__result-meta) {
    color: var(--grey-5);
    line-height: 1.8;
  }

  .pagefind-panel :global(.pagefind-ui__message) {
    padding: 1rem 0.25rem 0;
  }

  .pagefind-panel :global(mark) {
    border-radius: 0.35rem;
    background: color-mix(in srgb, var(--color-yellow) 30%, transparent);
    color: inherit;
    padding: 0.05rem 0.25rem;
  }

  .pagefind-panel :global(.pagefind-ui__button) {
    border-radius: 999px;
    border-color: color-mix(in srgb, var(--grey-4) 25%, transparent);
    background: color-mix(in srgb, var(--grey-0) 82%, transparent);
    color: var(--text-color);
    transition:
      transform 0.2s ease,
      border-color 0.2s ease,
      color 0.2s ease,
      background 0.2s ease;
  }

  .pagefind-panel :global(.pagefind-ui__button:hover),
  .pagefind-panel :global(.pagefind-ui__button:focus-visible) {
    transform: translateY(-1px);
    border-color: color-mix(in srgb, var(--color-red) 28%, var(--grey-4));
    background: color-mix(in srgb, var(--color-red-a1) 60%, var(--grey-1));
    color: var(--color-red);
    outline: none;
  }

  @media (max-width: 1023px) {
    .search-shell {
      padding-top: 4rem;
    }

    .pagefind-panel {
      min-height: min(30rem, calc(100vh - 5rem));
      max-height: calc(100vh - 5rem);
      border-radius: 1rem;
    }

    .search-panel__body {
      padding: 0.875rem;
      padding-top: 3.5rem;
    }
  }

  @media (max-width: 768px) {
    .search-shell {
      padding: 3.5rem 0.75rem 0.75rem;
    }

    .search-overlay {
      backdrop-filter: blur(10px) saturate(125%);
    }

    .pagefind-panel {
      min-height: calc(100vh - 4.25rem);
      max-height: calc(100vh - 4.25rem);
      width: 100%;
      border-radius: 0.9rem;
    }

    .search-panel__ornament {
      left: 1rem;
      top: 1rem;
    }

    .search-panel__glow {
      width: 3rem;
    }

    .search-panel__close {
      right: 0.75rem;
      top: 0.75rem;
      height: 2.5rem;
      width: 2.5rem;
    }

    .search-panel__body {
      padding-top: 3.25rem;
    }

    .pagefind-panel :global(.pagefind-ui__search-input) {
      min-height: 3.25rem;
      font-size: 0.95rem;
    }

    .pagefind-panel :global(.pagefind-ui__result) {
      padding: 0.9rem 1rem;
    }
  }
</style>
