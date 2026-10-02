interface GazeDelegate {
  subdelegates: {
    at(index: number): {
      update(): void;
      getLive2DManager(): { onDrag(x: number, y: number): void };
    };
  };
  initializeSubdelegates(): void;
  onMouseMove(event: MouseEvent): void;
  onMouseEnd(event: MouseEvent): void;
  transformOffset(event: MouseEvent): { x: number; y: number };
}

/** 鼠标静止时，切页入场动画仍会移动画布；每帧按最后的视口坐标刷新视线。 */
export function installLive2DGazeRefresh(prototype: GazeDelegate) {
  const pointers = new WeakMap<GazeDelegate, MouseEvent>();
  // oxlint-disable-next-line typescript/unbound-method -- 下方通过 call 绑定实际实例。
  const move = prototype.onMouseMove;
  prototype.onMouseMove = function (event) {
    pointers.set(this, event);
    move.call(this, event);
  };
  // oxlint-disable-next-line typescript/unbound-method -- 下方通过 call 绑定实际实例。
  const end = prototype.onMouseEnd;
  prototype.onMouseEnd = function (event) {
    if (event.relatedTarget !== null) return;
    pointers.delete(this);
    end.call(this, event);
  };
  // oxlint-disable-next-line typescript/unbound-method -- 下方通过 call 绑定实际实例。
  const initialize = prototype.initializeSubdelegates;
  prototype.initializeSubdelegates = function () {
    initialize.call(this);
    const delegate = this.subdelegates.at(0);
    const update = delegate.update.bind(delegate);
    delegate.update = () => {
      const pointer = pointers.get(this);
      if (pointer) {
        const { x, y } = this.transformOffset(pointer);
        // 只更新视线，不重放鼠标点击、碰触或动作。
        delegate.getLive2DManager().onDrag(x, y);
      }
      update();
    };
  };
}
