import { describe, expect, it, vi } from "vitest";
import { installCubism2ContextReset } from "./live2d-context";

describe("Cubism 2 上下文切换", () => {
  it("首次加载和同一画布换装保留缓存，切回新画布先释放旧遮罩", () => {
    const contexts = new Map<number, object>();
    const dispose = vi.fn(() => contexts.clear());
    const core = {
      getGL: (index: number) => contexts.get(index),
      setGL: vi.fn((context: object, index = 0) => {
        contexts.set(index, context);
      }),
      dispose,
    };
    installCubism2ContextReset(core);
    const first = { canvas: "旧画布" };
    const second = { canvas: "新画布" };

    core.setGL(first);
    core.setGL(first);
    expect(dispose).not.toHaveBeenCalled();

    core.setGL(second);
    expect(dispose).toHaveBeenCalledOnce();
    expect(contexts.get(0)).toBe(second);
  });
});
