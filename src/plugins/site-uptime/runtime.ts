export interface SiteUptimeRuntimeOptions {
  siteCreatedAt: string;
  prefixText?: string;
}

const FOOTER_STATUS_SELECTOR = "#footer .status";

function formatUptime(createdAt: Date, prefixText: string): string {
  const diff = Math.max(0, Date.now() - createdAt.getTime());
  const totalSeconds = Math.floor(diff / 1000);
  const totalMinutes = Math.floor(totalSeconds / 60);
  const totalHours = Math.floor(totalMinutes / 60);
  const totalDays = Math.floor(totalHours / 24);

  // 与参考项目保持一致：按 30 天折算一个月，并保留后续的日、时、分、秒。
  const months = Math.floor(totalDays / 30);
  const days = totalDays % 30;
  const hours = totalHours % 24;
  const minutes = totalMinutes % 60;
  const seconds = totalSeconds % 60;

  return `${prefixText}${months}个月${days}天${hours}小时${minutes}分钟${seconds}秒`;
}

function mountUptime(options: SiteUptimeRuntimeOptions): void {
  const target = document.querySelector(FOOTER_STATUS_SELECTOR);
  if (!target || target.querySelector(":scope > .site-uptime")) {
    return;
  }

  const container = document.createElement("div");
  container.className = "site-uptime";
  container.style.cssText = "margin: 1rem 0 0.5rem; font-size: 0.9em;";

  const createdAt = new Date(options.siteCreatedAt);
  const prefixText = options.prefixText ?? "本站点已经存在了";
  const update = () => {
    container.textContent = formatUptime(createdAt, prefixText);
  };

  update();
  const intervalId = window.setInterval(update, 1000);

  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.removedNodes) {
        if (node === container) {
          window.clearInterval(intervalId);
          observer.disconnect();
          return;
        }
      }
    }
  });

  queueMicrotask(() => {
    if (container.parentNode) {
      observer.observe(container.parentNode, { childList: true });
    }
  });

  target.appendChild(container);
}

export function init(options: SiteUptimeRuntimeOptions): void {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return;
  }

  const start = () => mountUptime(options);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
}
