<svelte:options customElement="image-zoom" />

<script lang="ts">
  import { onDestroy, onMount } from "svelte";

  let container = $state<HTMLElement | null>(null);
  let isOpen = $state(false);
  let previewSrc = $state("");
  let previewAlt = $state("");

  let cleanupImageListeners: (() => void) | null = null;
  let cleanupSlotListener: (() => void) | null = null;
  let closeTimer: ReturnType<typeof setTimeout> | null = null;
  let isClosing = $state(false);

  const CLOSE_ANIMATION_MS = 220;

  const restoreBodyScroll = () => {
    if (typeof document !== "undefined") {
      document.body.style.overflow = "";
    }
  };

  const finalizeClosePreview = () => {
    isOpen = false;
    isClosing = false;
    previewSrc = "";
    previewAlt = "";

    if (closeTimer) {
      clearTimeout(closeTimer);
      closeTimer = null;
    }

    restoreBodyScroll();
  };

  const requestClosePreview = () => {
    if (!isOpen || isClosing) {
      return;
    }

    const closeAnimationMs =
      typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? 0
        : CLOSE_ANIMATION_MS;

    if (closeAnimationMs === 0) {
      finalizeClosePreview();
      return;
    }

    isClosing = true;

    if (closeTimer) {
      clearTimeout(closeTimer);
    }

    closeTimer = setTimeout(() => {
      finalizeClosePreview();
    }, closeAnimationMs);
  };

  const openPreview = (image: HTMLImageElement, event?: Event) => {
    event?.preventDefault();
    event?.stopPropagation();

    const source = image.currentSrc || image.src;
    if (!source) {
      return;
    }

    if (closeTimer) {
      clearTimeout(closeTimer);
      closeTimer = null;
    }

    isClosing = false;
    previewSrc = source;
    previewAlt = image.alt || "";
    isOpen = true;

    if (typeof document !== "undefined") {
      document.body.style.overflow = "hidden";
    }
  };

  const bindImage = () => {
    cleanupImageListeners?.();
    cleanupImageListeners = null;

    const slot = container?.querySelector("slot") as HTMLSlotElement | null;
    const assignedElements = slot?.assignedElements({ flatten: true }) ?? [];
    const image = assignedElements.find((el) => el.tagName === "IMG") as
      | HTMLImageElement
      | undefined;

    if (!image) {
      return;
    }

    image.classList.add("image-zoom-trigger");

    const hasRole = image.hasAttribute("role");
    const hasTabindex = image.hasAttribute("tabindex");

    if (!hasRole) {
      image.setAttribute("role", "button");
    }

    if (!hasTabindex) {
      image.setAttribute("tabindex", "0");
    }

    const onImageClick = (event: MouseEvent) => {
      openPreview(image, event);
    };

    const onImageKeydown = (event: KeyboardEvent) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openPreview(image, event);
      }
    };

    image.addEventListener("click", onImageClick);
    image.addEventListener("keydown", onImageKeydown);

    cleanupImageListeners = () => {
      image.removeEventListener("click", onImageClick);
      image.removeEventListener("keydown", onImageKeydown);
      image.classList.remove("image-zoom-trigger");

      if (!hasRole) {
        image.removeAttribute("role");
      }

      if (!hasTabindex) {
        image.removeAttribute("tabindex");
      }
    };
  };

  const handleOverlayClick = (event: MouseEvent) => {
    const target = event.target as HTMLElement | null;
    if (target?.closest(".image-zoom-content")) {
      return;
    }

    requestClosePreview();
  };

  const handleWindowKeydown = (event: KeyboardEvent) => {
    if (event.key === "Escape" && isOpen) {
      requestClosePreview();
    }
  };

  onMount(() => {
    bindImage();

    const slot = container?.querySelector("slot") as HTMLSlotElement | null;
    if (slot) {
      const onSlotChange = () => {
        bindImage();
      };

      slot.addEventListener("slotchange", onSlotChange);
      cleanupSlotListener = () => {
        slot.removeEventListener("slotchange", onSlotChange);
      };
    }

    if (typeof window !== "undefined") {
      window.addEventListener("keydown", handleWindowKeydown);
    }
  });

  onDestroy(() => {
    cleanupSlotListener?.();
    cleanupImageListeners?.();

    if (closeTimer) {
      clearTimeout(closeTimer);
      closeTimer = null;
    }

    if (typeof window !== "undefined") {
      window.removeEventListener("keydown", handleWindowKeydown);
    }

    restoreBodyScroll();
  });
</script>

<div bind:this={container} class="image-zoom-wrapper">
  <slot />
</div>

{#if isOpen}
  <div
    class="image-zoom-overlay {isClosing ? 'closing' : ''}"
    role="dialog"
    aria-modal="true"
    aria-label={previewAlt || "图片预览"}
    onclick={handleOverlayClick}
  >
    <button
      type="button"
      class="image-zoom-close"
      aria-label="关闭图片预览"
      onclick={requestClosePreview}
    >
      ×
    </button>
    <img
      class="image-zoom-content"
      src={previewSrc}
      alt={previewAlt}
      loading="eager"
      decoding="async"
      onclick={requestClosePreview}
    />
    {#if previewAlt}
      <p class="image-zoom-caption">{previewAlt}</p>
    {/if}
  </div>
{/if}

<style>
  :global(image-zoom) {
    display: block;
    margin: 1.2rem auto;
    width: fit-content;
    max-width: 100%;
  }

  .image-zoom-wrapper {
    display: block;
    max-width: 100%;
  }

  :global(image-zoom .image-zoom-trigger) {
    cursor: zoom-in;
    transform-origin: center;
    transition:
      filter 0.2s ease,
      transform 0.2s ease;
  }

  :global(image-zoom .image-zoom-trigger:hover) {
    filter: brightness(0.96);
    transform: translateY(-1px) scale(1.01);
  }

  .image-zoom-overlay {
    position: fixed;
    inset: 0;
    z-index: var(--z-fullscreen);
    display: grid;
    place-items: center;
    padding: 2rem 1rem;
    box-sizing: border-box;
    background: var(--codeblock-overlay-bg, rgba(8, 10, 16, 0.72));
    backdrop-filter: blur(0.35rem);
    animation: image-zoom-fade-in 220ms ease forwards;
  }

  .image-zoom-overlay.closing {
    animation: image-zoom-fade-out 220ms ease forwards;
  }

  .image-zoom-content {
    margin: 0;
    max-width: min(92vw, 1100px);
    max-height: 86vh;
    object-fit: contain;
    border-radius: 0.5rem;
    box-shadow: 0 0.75rem 2rem var(--grey-9-a15);
    cursor: zoom-out;
    animation: image-zoom-scale-in 220ms cubic-bezier(0.22, 0.61, 0.36, 1) forwards;
  }

  .image-zoom-overlay.closing .image-zoom-content {
    animation: image-zoom-scale-out 220ms cubic-bezier(0.4, 0, 0.2, 1) forwards;
  }

  .image-zoom-close {
    position: absolute;
    top: 1rem;
    right: 1rem;
    width: 2.4rem;
    height: 2.4rem;
    border: 0;
    border-radius: 50%;
    background: rgba(17, 25, 40, 0.58);
    color: #fff;
    font-size: 1.5rem;
    line-height: 1;
    cursor: pointer;
    transition:
      background-color 0.2s ease,
      transform 0.2s ease;
    animation: image-zoom-ui-in 220ms ease forwards;
  }

  .image-zoom-overlay.closing .image-zoom-close {
    animation: image-zoom-ui-out 220ms ease forwards;
  }

  .image-zoom-close:hover {
    background: rgba(17, 25, 40, 0.8);
    transform: scale(1.06);
  }

  .image-zoom-caption {
    margin: 0.8rem 0 0;
    font-size: 0.9rem;
    color: var(--grey-1);
    text-align: center;
    max-width: min(92vw, 1100px);
    line-height: 1.5;
    animation: image-zoom-ui-in 220ms ease forwards;
  }

  .image-zoom-overlay.closing .image-zoom-caption {
    animation: image-zoom-ui-out 220ms ease forwards;
  }

  @keyframes image-zoom-fade-in {
    from {
      opacity: 0;
    }

    to {
      opacity: 1;
    }
  }

  @keyframes image-zoom-fade-out {
    from {
      opacity: 1;
    }

    to {
      opacity: 0;
    }
  }

  @keyframes image-zoom-scale-in {
    from {
      opacity: 0;
      transform: scale(0.94) translateY(10px);
    }

    to {
      opacity: 1;
      transform: scale(1) translateY(0);
    }
  }

  @keyframes image-zoom-scale-out {
    from {
      opacity: 1;
      transform: scale(1) translateY(0);
    }

    to {
      opacity: 0;
      transform: scale(0.94) translateY(10px);
    }
  }

  @keyframes image-zoom-ui-in {
    from {
      opacity: 0;
      transform: translateY(-6px);
    }

    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes image-zoom-ui-out {
    from {
      opacity: 1;
      transform: translateY(0);
    }

    to {
      opacity: 0;
      transform: translateY(-6px);
    }
  }

  @media (prefers-reduced-motion: reduce) {
    :global(image-zoom .image-zoom-trigger),
    .image-zoom-close {
      transition: none;
    }

    .image-zoom-overlay,
    .image-zoom-overlay.closing,
    .image-zoom-content,
    .image-zoom-overlay.closing .image-zoom-content,
    .image-zoom-caption,
    .image-zoom-overlay.closing .image-zoom-caption,
    .image-zoom-overlay.closing .image-zoom-close {
      animation: none;
    }
  }
</style>
