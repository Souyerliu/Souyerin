// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { installLive2DGazeRefresh } from "./live2d-gaze";

function createRuntime() {
  const drag = vi.fn();
  const render = vi.fn();
  const move = vi.fn();
  let canvasTop = 500;
  const delegate = { update: render, getLive2DManager: () => ({ onDrag: drag }) };
  const runtime = {
    subdelegates: { at: () => delegate },
    initializeSubdelegates: vi.fn(),
    onMouseMove: move,
    onMouseEnd: vi.fn(),
    transformOffset: (event: MouseEvent) => ({ x: event.clientX, y: canvasTop - event.clientY }),
  };
  installLive2DGazeRefresh(runtime);
  runtime.initializeSubdelegates();
  return { runtime, delegate, drag, render, move, setTop: (top: number) => (canvasTop = top) };
}

describe("Live2D 切页后的视线刷新", () => {
  it("鼠标静止时跟随画布入场、持久化搬移和最终位置，不重放鼠标动作", () => {
    const { runtime, delegate, drag, render, move, setTop } = createRuntime();
    runtime.onMouseMove(new MouseEvent("mousemove", { clientX: 700, clientY: 150 }));
    delegate.update();
    expect(drag).toHaveBeenLastCalledWith(700, 350);
    setTop(1000);
    delegate.update();
    expect(drag).toHaveBeenLastCalledWith(700, 850);
    setTop(500);
    delegate.update();
    expect(drag).toHaveBeenLastCalledWith(700, 350);
    expect(move).toHaveBeenCalledOnce();
    expect(render).toHaveBeenCalledTimes(3);
  });

  it("内部元素边界保留光标，离开窗口后停止刷新以保留回正行为", () => {
    const { runtime, delegate, drag } = createRuntime();
    runtime.onMouseMove(new MouseEvent("mousemove", { clientY: 150 }));
    runtime.onMouseEnd(new MouseEvent("mouseout", { relatedTarget: document.createElement("a") }));
    delegate.update();
    expect(drag).toHaveBeenCalledTimes(1);
    runtime.onMouseEnd(new MouseEvent("mouseout"));
    delegate.update();
    expect(drag).toHaveBeenCalledTimes(1);
  });

  it("未收到光标时不覆盖模型默认动作，不同实例互不影响", () => {
    const first = createRuntime();
    const second = createRuntime();
    first.runtime.onMouseMove(new MouseEvent("mousemove"));
    first.delegate.update();
    second.delegate.update();
    expect(first.drag).toHaveBeenCalledOnce();
    expect(second.drag).not.toHaveBeenCalled();
    expect(second.render).toHaveBeenCalledOnce();
  });
});
