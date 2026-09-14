export function haveSameCategories(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((category, index) => category === right[index]);
}

/**
 * Selects a centered window from posts that are already in display order.
 * Near either end of the series, the window shifts to keep `limit` neighbors.
 */
export function selectSurroundingPosts<T extends { id: string }>(
  posts: readonly T[],
  currentId: string,
  limit: number,
): T[] {
  const currentIndex = posts.findIndex((post) => post.id === currentId);
  const normalizedLimit = Math.max(0, Math.floor(limit));

  if (currentIndex < 0 || normalizedLimit === 0) {
    return [];
  }

  const windowSize = normalizedLimit + 1;
  let start = Math.max(0, currentIndex - Math.floor(normalizedLimit / 2));
  let end = Math.min(posts.length, start + windowSize);

  start = Math.max(0, end - windowSize);
  end = Math.min(posts.length, start + windowSize);

  return posts.slice(start, end).filter((post) => post.id !== currentId);
}
