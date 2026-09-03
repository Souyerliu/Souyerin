import { cleanup, fireEvent, render } from "@solidjs/testing-library";
import { afterEach, describe, expect, it, vi } from "vitest";

import SidebarContents from "./SidebarContents";

class IntersectionObserverStub {
  disconnect(): void {}
  observe(): void {}
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("SidebarContents", () => {
  it("prevents ClientRouter from jumping before starting smooth scrolling", () => {
    vi.stubGlobal("IntersectionObserver", IntersectionObserverStub);
    const scrollIntoView = vi.fn();
    const heading = document.createElement("h2");
    heading.id = "details";
    heading.scrollIntoView = scrollIntoView;
    document.body.append(heading);

    let routerSawPreventedClick = false;
    const clientRouterListener = (event: MouseEvent) => {
      if ((event.target as Element).closest(".toc-link")) {
        routerSawPreventedClick = event.defaultPrevented;
      }
    };
    document.addEventListener("click", clientRouterListener);

    const { getByRole } = render(() => (
      <SidebarContents toc={[{ id: "details", text: "Details", level: 2 }]} />
    ));
    fireEvent.click(getByRole("link", { name: "Details" }));

    expect(routerSawPreventedClick).toBe(true);
    expect(scrollIntoView).toHaveBeenCalledWith({ behavior: "smooth" });

    document.removeEventListener("click", clientRouterListener);
    heading.remove();
  });
});
