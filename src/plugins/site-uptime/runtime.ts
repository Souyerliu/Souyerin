import type { PluginInitFunction } from "@hyacine/helper/runtime";
import { getInjectPointSelector } from "@hyacine/helper/runtime";

interface SiteUptimeRuntimeOptions {
  siteCreatedAt: string;
  prefixText: string;
}

function calculateUptime(createdAt: Date, prefixText: string): string {
  const diff = Date.now() - createdAt.getTime();

  if (diff < 0) {
    return `${prefixText} 0秒`;
  }

  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  const months = Math.floor(days / 30);
  const years = Math.floor(months / 12);

  const remainingMonths = months % 12;
  const remainingDays = days % 30;
  const remainingHours = hours % 24;
  const remainingMinutes = minutes % 60;
  const remainingSeconds = seconds % 60;
  const parts: string[] = [];

  if (years > 0) parts.push(`${years}年`);
  if (remainingMonths > 0) parts.push(`${remainingMonths}个月`);
  if (remainingDays > 0) parts.push(`${remainingDays}天`);
  if (remainingHours > 0) parts.push(`${remainingHours}小时`);
  if (remainingMinutes > 0) parts.push(`${remainingMinutes}分钟`);
  if (remainingSeconds > 0 || parts.length === 0) {
    parts.push(`${remainingSeconds}秒`);
  }

  return `${prefixText} ${parts.join("")}`;
}

function createUptimeElement(options: SiteUptimeRuntimeOptions): HTMLElement {
  const container = document.createElement("div");
  container.className = "site-uptime";
  container.style.cssText = "margin: 0.5rem 0; font-size: 0.9em;";

  const createdAt = new Date(options.siteCreatedAt);
  const updateUptime = () => {
    container.textContent = calculateUptime(createdAt, options.prefixText);
  };

  updateUptime();
  const intervalId = window.setInterval(updateUptime, 1000);

  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.removedNodes) {
        if (node === container) {
          window.clearInterval(intervalId);
          observer.disconnect();
        }
      }
    }
  });

  queueMicrotask(() => {
    if (container.parentNode) {
      observer.observe(container.parentNode, { childList: true });
    }
  });

  return container;
}

function mountUptime(options: SiteUptimeRuntimeOptions): void {
  const selector = getInjectPointSelector("footer-status");
  const targetElement = document.querySelector(selector);
  if (!targetElement || targetElement.querySelector(":scope > .site-uptime")) {
    return;
  }

  const element = createUptimeElement(options);
  targetElement.appendChild(element);
}

export const init: PluginInitFunction<SiteUptimeRuntimeOptions> = (options) => {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => mountUptime(options), {
      once: true,
    });
    return;
  }

  mountUptime(options);
};
