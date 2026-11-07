import { describe, expect, it, vi } from "vitest";
import { installLive2DPropsFilter } from "./live2d-props";

function fixture() {
  const parts = {
    ids: ["Part23", "Part47", "Part4"],
    opacities: new Float32Array([1, 1, 1]),
  };
  const updateCore = vi.fn();
  const cubism = { getModel: () => ({ parts }), update: updateCore };
  let loaded = false;
  const model = {
    getModel: () => (loaded ? cubism : null),
    update: vi.fn(() => {
      if (!loaded) return;
      // 模拟动作写入部件透明度，过滤必须在核心计算之前覆盖它。
      parts.opacities.fill(1);
      cubism.update();
    }),
  };
  const delegate = {
    changeModel: vi.fn((_path: string) => {}),
    subdelegates: { at: () => ({ getLive2DManager: () => ({ _models: { at: () => model } }) }) },
  };
  installLive2DPropsFilter(delegate);
  return { delegate, model, parts, updateCore, load: () => (loaded = true) };
}

describe("Yuno 表演设备隐藏", () => {
  it.each(["live", "birthday2627"])("%s 异步加载后每帧只隐藏设备", (costume) => {
    const { delegate, model, parts, updateCore, load } = fixture();
    delegate.changeModel(`/live2d-models/model/yuno/live2d_yuno_015_${costume}_01/index.json`);
    model.update();
    expect(updateCore).not.toHaveBeenCalled();
    load();
    updateCore.mockImplementation(() => {
      expect([...parts.opacities]).toEqual([1, 0, 1]);
    });
    model.update();
    model.update();
    expect(updateCore).toHaveBeenCalledTimes(2);
  });

  it("其他服装保留原始更新方法及全部部件", () => {
    const { delegate, model, parts, load } = fixture();
    const original = model.update;
    delegate.changeModel("/live2d-models/model/yuno/adv_live2d_yuno_015_virtual_01/index.json");
    load();
    model.update();
    expect(model.update).toBe(original);
    expect([...parts.opacities]).toEqual([1, 1, 1]);
  });
});
