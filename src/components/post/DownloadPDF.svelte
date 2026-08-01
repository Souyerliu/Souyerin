<script lang="ts">
  import {
    fetchAllStylesheets,
    resolveImageUrls,
    inlineEssentialStyles,
    buildPrintHTML,
  } from "@/toolkit/generatePrintHTML";

  interface Props {
    title: string;
  }

  let { title }: Props = $props();

  let isGenerating = $state(false);
  let error = $state<string | null>(null);

  const generatePDF = async () => {
    if (isGenerating) return;
    isGenerating = true;
    error = null;

    let printWindow: Window | null = null;

    try {
      const article = document.querySelector("article.post.block");
      if (!article) throw new Error("未找到文章内容");

      const allCSS = await fetchAllStylesheets();

      // 每次独立 deep-clone，不修改原始 DOM
      const clone = article.cloneNode(true) as HTMLElement;
      inlineEssentialStyles(article, clone);
      resolveImageUrls(clone);

      const removeSel = [".pdf-download-area", "ai-similar-posts", ".ai-summary-card"];
      for (const sel of removeSel) {
        clone.querySelectorAll(sel).forEach((e) => e.remove());
      }

      const html = buildPrintHTML(title, clone, allCSS);

      // about:blank + document.write 替代 blob URL。
      // 每次用唯一窗口名避免浏览器重用旧窗口。
      const winName = `print-${Date.now()}`;
      printWindow = window.open("about:blank", winName);
      if (!printWindow) {
        throw new Error("弹窗被浏览器拦截，请允许本站弹出窗口后重试");
      }

      printWindow.document.open();
      printWindow.document.write(html);
      printWindow.document.close();

      // 运行时：对溢出块公式按需缩放（双 rAF 等布局完成）
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          const bodyEl = printWindow!.document.body;
          const cs = getComputedStyle(bodyEl);
          const baseFont = parseFloat(cs.fontSize);
          const padL = parseFloat(cs.paddingLeft) || 0;
          const padR = parseFloat(cs.paddingRight) || 0;
          const maxW = bodyEl.clientWidth - padL - padR;

          printWindow!.document
            .querySelectorAll<HTMLElement>(".katex-display")
            .forEach((el) => {
              const katex = el.querySelector<HTMLElement>(">.katex");
              if (!katex) return;
              // 用 scrollWidth 更真实反映溢出（offsetWidth 可能被裁剪）
              const realW = Math.max(katex.scrollWidth, katex.offsetWidth);
              if (realW <= maxW) return;
              // 缩放比例，给 10% 安全余量
              const ratio = (maxW / realW) * 0.9;
              // 下限 40%，上限 100%
              const clamped = Math.max(0.4, Math.min(1, ratio));
              el.style.fontSize = `${(clamped * baseFont).toFixed(1)}px`;
              katex.style.fontSize = "inherit";
            });
        });
      });

      // 等图片全部加载完成后再打印（最长 8 秒）
      const printed = new Promise<void>((resolve) => {
        const win = printWindow!;
        const tryPrint = () => {
          const images = Array.from(win.document.images);
          const allLoaded = images.every(
            (img) => (img as HTMLImageElement).complete,
          );
          if (allLoaded || images.length === 0) {
            win.print();
            resolve();
          } else {
            setTimeout(tryPrint, 200);
          }
        };
        setTimeout(tryPrint, 400);
      });

      const timeout = new Promise<void>((resolve) => {
        setTimeout(() => {
          printWindow?.print();
          resolve();
        }, 8000);
      });

      await Promise.race([printed, timeout]);

      // 打印对话框关闭后清理
      setTimeout(() => {
        printWindow?.close();
        isGenerating = false;
      }, 800);
    } catch (e) {
      error = e instanceof Error ? e.message : "PDF 生成失败，请稍后重试";
      console.error("[DownloadPDF] 生成失败:", e);
      isGenerating = false;
      try { printWindow?.close(); } catch { /* empty */ }
    }
  };
</script>

<div style="display:flex;flex-direction:column;align-items:center;margin-top:2rem;padding-top:1.5rem">
  <button
    type="button"
    style="display:inline-flex;align-items:center;gap:0.5rem;padding:0.625rem 1.5rem;border:1px solid;border-color:var(--primary-color);color:var(--primary-color);border-radius:0.5rem;background:transparent;cursor:pointer"
    onclick={generatePDF}
    disabled={isGenerating}
    title="下载文章为 PDF 文件"
  >
    {#if isGenerating}
      <span class="i-ri-loader-4-line animate-spin"></span>
      <span>正在准备打印...</span>
    {:else}
      <span class="i-ri-file-download-line"></span>
      <span>下载 PDF</span>
    {/if}
  </button>

  {#if error}
    <p style="display:flex;align-items:center;gap:0.375rem;color:var(--color-red-a11);margin-top:0.625rem">
      <span class="i-ri-error-warning-line"></span>
      {error}
    </p>
  {/if}
</div>
