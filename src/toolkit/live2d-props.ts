interface PerformanceModel {
  update(): void;
  getModel(): {
    update(): void;
    getModel(): {
      parts: { ids: string[]; opacities: Float32Array };
    };
  } | null;
}

interface PropsDelegate {
  changeModel(path: string): void;
  subdelegates: {
    at(index: number): {
      getLive2DManager(): { _models: { at(index: number): PerformanceModel } };
    };
  };
}

/** 两套 Yuno 表演服装的 Part47 是整套 DJ 设备，保留人物及佩戴的耳机。 */
export function installLive2DPropsFilter(prototype: PropsDelegate) {
  // oxlint-disable-next-line typescript/unbound-method -- 下方通过 call 绑定实际实例。
  const changeModel = prototype.changeModel;
  prototype.changeModel = function (path) {
    changeModel.call(this, path);
    if (!/\/yuno\/live2d_yuno_015_(?:live|birthday2627)_01\/index\.json(?:[?#].*)?$/.test(path)) {
      return;
    }
    // oxlint-disable-next-line no-underscore-dangle -- CDN 运行时仅通过 _models 暴露模型列表。
    const model = this.subdelegates.at(0).getLive2DManager()._models.at(0);
    const update = model.update.bind(model);
    let installed = false;
    model.update = () => {
      // 模型异步加载；在首个可用帧安装，隐藏发生在动作、物理计算之后的核心更新前。
      const cubism = model.getModel();
      if (cubism && !installed) {
        const parts = cubism.getModel().parts;
        const index = parts.ids.indexOf("Part47");
        if (index >= 0) {
          const updateCore = cubism.update.bind(cubism);
          cubism.update = () => {
            parts.opacities[index] = 0;
            updateCore();
          };
        }
        installed = true;
      }
      update();
    };
  };
}
