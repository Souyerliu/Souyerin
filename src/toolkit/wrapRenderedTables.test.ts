import { describe, expect, it } from "vitest";

import { wrapRenderedTables } from "./wrapRenderedTables";

describe("wrapRenderedTables", () => {
  it("wraps a rendered Markdown table", () => {
    const html = "<p>before</p><table><tbody><tr><td>cell</td></tr></tbody></table>";

    expect(wrapRenderedTables(html)).toBe(
      '<p>before</p><div class="table-container"><table><tbody><tr><td>cell</td></tr></tbody></table></div>',
    );
  });

  it("does not wrap an already wrapped table twice", () => {
    const html =
      '<div class="table-container"><table><tbody><tr><td>cell</td></tr></tbody></table></div>';

    expect(wrapRenderedTables(html)).toBe(html);
  });
});
