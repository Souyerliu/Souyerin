import type { ImageMetadata } from "astro";

const coverModules = import.meta.glob<{ default: ImageMetadata }>(
  "../assets/images/*.{jpg,jpeg,png,webp,avif}",
  { eager: true },
);

const numericCoverEntries = Object.entries(coverModules)
  .flatMap(([path, module]) => {
    const stem = path.match(/\/(\d+)\.[^/]+$/)?.[1];
    return stem ? [{ path, stem, image: module.default }] : [];
  })
  .sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }));

export const covers = numericCoverEntries.map(({ image }) => image);

export const coverPresets: Record<string, ImageMetadata> = Object.fromEntries(
  numericCoverEntries.map(({ stem, image }) => [`cover-${stem}`, image]),
);
