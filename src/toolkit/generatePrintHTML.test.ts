// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { fitDisplayMath, PRINT_CONTENT_WIDTH } from "./generatePrintHTML";

/** jsdom 不做布局，clientWidth/scrollWidth 恒为 0，这里手动打桩模拟真实测量值 */
function createDisplay(
  clientWidth: number,
  scrollWidth: number,
  innerScrollWidth = 0,
): HTMLElement {
  const display = document.createElement("div");
  display.className = "katex-display";
  const inner = document.createElement("span");
  inner.className = "katex";
  display.append(inner);

  Object.defineProperty(display, "clientWidth", { value: clientWidth });
  Object.defineProperty(display, "scrollWidth", { value: scrollWidth });
  Object.defineProperty(inner, "scrollWidth", { value: innerScrollWidth });
  return display;
}

describe("fitDisplayMath", () => {
  it("公式未超宽时不改字号", () => {
    const root = document.createElement("div");
    root.append(createDisplay(700, 700));
    fitDisplayMath(root);

    const display = root.querySelector<HTMLElement>(".katex-display")!;
    expect(display.style.fontSize).toBe("");
  });

  it("1px 以内的测量误差不触发缩放", () => {
    const root = document.createElement("div");
    root.append(createDisplay(700, 701));
    fitDisplayMath(root);

    expect(root.querySelector<HTMLElement>(".katex-display")!.style.fontSize).toBe("");
  });

  it("超宽公式按可用宽度与真实宽度之比缩小", () => {
    const root = document.createElement("div");
    root.append(createDisplay(700, 1400));
    fitDisplayMath(root);

    expect(root.querySelector<HTMLElement>(".katex-display")!.style.fontSize).toBe("50%");
  });

  it("缩放不会低于下限", () => {
    const root = document.createElement("div");
    root.append(createDisplay(100, 1000));
    fitDisplayMath(root, 0.45);

    expect(root.querySelector<HTMLElement>(".katex-display")!.style.fontSize).toBe("45%");
  });

  it("以内层 .katex 的真实宽度为准（外层被 max-width 夹住时）", () => {
    const root = document.createElement("div");
    const display = createDisplay(700, 700, 1400);
    root.append(display);
    fitDisplayMath(root);

    expect(display.style.fontSize).toBe("50%");
  });

  it("统一把 overflow 置为 visible 以消除预览滚动条", () => {
    const root = document.createElement("div");
    root.append(createDisplay(700, 700), createDisplay(700, 1400));
    fitDisplayMath(root);

    for (const display of root.querySelectorAll<HTMLElement>(".katex-display")) {
      expect(display.style.overflow).toBe("visible");
    }
  });

  it("宽度为 0（元素不可见）时跳过缩放", () => {
    const root = document.createElement("div");
    root.append(createDisplay(0, 0));
    fitDisplayMath(root);

    expect(root.querySelector<HTMLElement>(".katex-display")!.style.fontSize).toBe("");
  });
});

describe("PRINT_CONTENT_WIDTH", () => {
  it("等于 A4 纸宽减去左右页边距", () => {
    expect(PRINT_CONTENT_WIDTH).toBe("186mm");
  });
});
