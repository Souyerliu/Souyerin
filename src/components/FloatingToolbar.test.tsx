import { cleanup, render } from "@solidjs/testing-library";
import { afterEach, describe, expect, it } from "vitest";

import FloatingToolbar from "./FloatingToolbar";

afterEach(() => {
  cleanup();
});

describe("FloatingToolbar", () => {
  it("在播放器水合前也渲染 NyxPlayer 控制按钮", () => {
    const { container } = render(() => <FloatingToolbar />);

    expect(container.querySelector("#nyx-show-btn")).not.toBeNull();
    expect(container.querySelector("#nyx-play-btn")).not.toBeNull();
  });
});
