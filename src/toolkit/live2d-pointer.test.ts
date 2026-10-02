// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { installLive2DPointerTracking } from "./live2d-pointer";

function createPointers() {
  const canvas = document.createElement("canvas");
  canvas.width = 600;
  canvas.height = 600;
  const rect = vi
    .spyOn(canvas, "getBoundingClientRect")
    .mockReturnValue(new DOMRect(100, 400, 300, 300));
  const mouseEvent = vi.fn();
  const mouseEnd = vi.fn();
  const cubism2 = { mouseEvent };
  const cubism3 = {
    subdelegates: {
      at: () => ({
        getCanvas: () => canvas,
        _view: { transformViewX: (x: number) => x, transformViewY: (y: number) => -y },
      }),
    },
    transformOffset: (_event: MouseEvent) => ({ x: 0, y: 0 }),
    onMouseEnd: mouseEnd,
  };
  installLive2DPointerTracking(cubism2, cubism3);
  return { cubism2, cubism3, canvas, rect, mouseEvent, mouseEnd };
}

describe("Live2D 光标跟随", () => {
  it("页面滚动距离不影响同一视口光标的视线方向", () => {
    const { cubism3 } = createPointers();
    const event = new MouseEvent("mousemove", { clientX: 250, clientY: 450 });
    const beforeScroll = cubism3.transformOffset(event);
    Object.defineProperty(event, "pageY", { value: 5450 });
    Object.defineProperty(event, "pageX", { value: 1250 });
    expect(cubism3.transformOffset(event)).toEqual(beforeScroll);
    expect(beforeScroll).toEqual({ x: 300, y: -100 });
  });

  it("使用当前画布尺寸和位置，不沿用旧的缩放与布局", () => {
    const { cubism3, canvas, rect } = createPointers();
    canvas.width = 300;
    canvas.height = 450;
    rect.mockReturnValue(new DOMRect(200, 300, 150, 150));
    expect(
      cubism3.transformOffset(new MouseEvent("mousemove", { clientX: 250, clientY: 350 })),
    ).toEqual({ x: 100, y: -150 });
    rect.mockReturnValue(new DOMRect());
    expect(cubism3.transformOffset(new MouseEvent("mousemove"))).toEqual({ x: 0, y: 0 });
  });

  it("经过页面元素不重置视线，真正离开窗口仍交回原有处理", () => {
    const { cubism2, cubism3, mouseEvent, mouseEnd } = createPointers();
    const internal = new MouseEvent("mouseout", { relatedTarget: document.createElement("a") });
    cubism2.mouseEvent(internal);
    cubism3.onMouseEnd(internal);
    expect(mouseEvent).not.toHaveBeenCalled();
    expect(mouseEnd).not.toHaveBeenCalled();
    const leave = new MouseEvent("mouseout");
    cubism2.mouseEvent(leave);
    cubism3.onMouseEnd(leave);
    expect(mouseEvent).toHaveBeenCalledWith(leave);
    expect(mouseEnd).toHaveBeenCalledWith(leave);
  });
});
