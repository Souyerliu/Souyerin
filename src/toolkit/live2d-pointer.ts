interface Cubism2Pointer {
  mouseEvent(event: MouseEvent): void;
}

interface Cubism3Pointer {
  subdelegates: {
    at(index: number): {
      getCanvas(): Pick<HTMLCanvasElement, "width" | "height" | "getBoundingClientRect">;
      _view: { transformViewX(x: number): number; transformViewY(y: number): number };
    };
  };
  transformOffset(event: MouseEvent): { x: number; y: number };
  onMouseEnd(event: MouseEvent): void;
}

/** 修正 CDN 库的滚动坐标混用，并避免经过普通元素边界时重置视线。 */
export function installLive2DPointerTracking(
  cubism2: Cubism2Pointer | undefined,
  cubism3: Cubism3Pointer | undefined,
) {
  if (cubism2) {
    // oxlint-disable-next-line typescript/unbound-method -- 原型方法在下面通过 call 绑定实际实例。
    const mouseEvent = cubism2.mouseEvent;
    cubism2.mouseEvent = function (event) {
      if (event.type === "mouseout" && event.relatedTarget !== null) return;
      mouseEvent.call(this, event);
    };
  }

  if (!cubism3) return;

  cubism3.transformOffset = function (event) {
    const delegate = this.subdelegates.at(0);
    const canvas = delegate.getCanvas();
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return { x: 0, y: 0 };
    // client 坐标和 DOMRect 同属视口；按实际画布比例换算，兼容缩放和高 DPI。
    const x = (event.clientX - rect.left) * (canvas.width / rect.width);
    const y = (event.clientY - rect.top) * (canvas.height / rect.height);
    // oxlint-disable-next-line no-underscore-dangle -- CDN 运行时仅通过 _view 暴露坐标变换。
    return { x: delegate._view.transformViewX(x), y: delegate._view.transformViewY(y) };
  };

  // oxlint-disable-next-line typescript/unbound-method -- 原型方法在下面通过 call 绑定实际实例。
  const mouseEnd = cubism3.onMouseEnd;
  cubism3.onMouseEnd = function (event) {
    if (event.relatedTarget !== null) return;
    mouseEnd.call(this, event);
  };
}
