/** Cubism 2 的遮罩缓存属于创建它的 WebGL 上下文，换画布时必须重建。 */
export function installCubism2ContextReset<Context>(core: {
  getGL(index: number): Context | undefined;
  setGL(context: Context, index?: number): void;
  dispose(): void;
}) {
  const originalSetGL = core.setGL.bind(core);
  core.setGL = function (context, index = 0) {
    const previousContext = core.getGL(index);
    if (previousContext && previousContext !== context) {
      core.dispose();
    }
    originalSetGL(context, index);
  };
}
