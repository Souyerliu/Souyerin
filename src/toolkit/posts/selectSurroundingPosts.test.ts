import { describe, expect, it } from "vitest";
import { haveSameCategories, selectSurroundingPosts } from "./selectSurroundingPosts";

const POSTS = Array.from({ length: 15 }, (_, index) => ({
  id: String(15 - index),
}));

describe("haveSameCategories", () => {
  it("requires the complete category path to match in order", () => {
    expect(haveSameCategories(["CS", "CS61B"], ["CS", "CS61B"])).toBe(true);
    expect(haveSameCategories(["CS", "CS61B"], ["Other", "CS61B"])).toBe(false);
    expect(haveSameCategories(["CS", "CS61B"], ["CS61B"])).toBe(false);
  });
});

describe("selectSurroundingPosts", () => {
  it("selects five posts on each side of a centered article", () => {
    expect(selectSurroundingPosts(POSTS.slice(4), "6", 10).map((post) => post.id)).toEqual([
      "11",
      "10",
      "9",
      "8",
      "7",
      "5",
      "4",
      "3",
      "2",
      "1",
    ]);
  });

  it("shifts the window near the newest and oldest ends", () => {
    expect(selectSurroundingPosts(POSTS, "15", 10).map((post) => post.id)).toEqual(
      POSTS.slice(1, 11).map((post) => post.id),
    );
    expect(selectSurroundingPosts(POSTS, "1", 10).map((post) => post.id)).toEqual(
      POSTS.slice(4, 14).map((post) => post.id),
    );
  });
});
