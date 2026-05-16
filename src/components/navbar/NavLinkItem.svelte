<script lang="ts">
  import NavItem from "./NavItem.svelte";

  interface Props {
    href?: string;
    text?: string;
    icon?: string | null;
    class?: string;
  }

  const {
    href = "#",
    text,
    icon = null,
    class: className = "",
  }: Props = $props();

  const iconClasses = $derived(
    icon ? `${icon} icon-nav text-xl vertical-text-bottom inline-block` : "",
  );

  const mergedClass = $derived(
    [className].filter(Boolean).join(" ")
  );

  // 判断是否为外链
  const isExternal = $derived(
    href.startsWith("http://") ||
    href.startsWith("https://")
  );
</script>

<NavItem class={mergedClass}>
  <a
    {href}
    target={isExternal ? "_blank" : undefined}
    rel={isExternal ? "noopener noreferrer" : undefined}
  >
    {#if icon}
      <div class={iconClasses}></div>
    {/if}
    {text}
  </a>
</NavItem>