import { createSignal, For, onCleanup, onMount } from "solid-js";

import { shuffle } from "es-toolkit";
import { currentLocale, t } from "@/i18n";
import { toPostHref } from "@/toolkit/posts/url";

interface Post {
  id: string;
  slug?: string;
  data: {
    title: string;
    description?: string;
  };
}

interface RecentCommentItem {
  nick: string;
  time: string;
  text: string;
  href: string;
  /** 评论所属页面路径（来源判定的依据） */
  path: string;
}

interface RecentCommentItemWithSite extends RecentCommentItem {
  site: "main" | "archive";
}

interface WidgetsProps {
  posts?: Post[];
  enableRandomPosts?: boolean;
  enableRecentComments?: boolean;
  recentCommentsLimit?: number;
  walineServerURL?: string;
  /**
   * 当前站点在“主站/副站”中的身份，用于标注“最新评论”的来源。
   * - "main"：主站
   * - "archive"：副站
   * - 未配置时按 "main" 处理
   */
  recentCommentsSiteRole?: "main" | "archive";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function safeString(value: unknown): string {
  if (value == null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  return JSON.stringify(value);
}

function formatDateTime(input: unknown): string {
  if (input == null) {
    return "";
  }

  const date = input instanceof Date ? input : new Date(safeString(input));

  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return date.toLocaleString(currentLocale, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** 线性剥离 HTML 标签（评论文本来源于不可控输入，正则 `<[^>]*>` 存在回溯风险） */
function stripHtmlTags(input: string): string {
  let out = "";
  let inTag = false;
  for (const ch of input) {
    if (ch === "<") {
      inTag = true;
    } else if (ch === ">") {
      inTag = false;
    } else if (!inTag) {
      out += ch;
    }
  }
  return out;
}

function toPlainText(input: unknown): string {
  if (input == null) {
    return "";
  }
  return stripHtmlTags(safeString(input)).replaceAll(/\s+/g, " ").trim();
}

function normalizePath(path: string): string {
  if (!path) {
    return "/";
  }

  if (/^https?:\/\//.test(path)) {
    return path;
  }

  return path.endsWith("/") ? path : `${path}/`;
}

/** 解码路径（Waline 记录的是浏览器 location.pathname，可能为百分号编码） */
function decodePath(input: string): string {
  try {
    return decodeURI(input);
  } catch {
    return input;
  }
}

/** 本站全部文章的页面路径集合（评论只产生于 /posts/... 文章页） */
function createOwnPostPaths(posts?: Post[]): Set<string> {
  return new Set(
    (posts ?? [])
      .map((post) => decodePath(toPostHref(post.slug || post.id)))
      .filter((path) => path.length > 0),
  );
}

/**
 * 判定评论来源（主站/副站）。
 * 两站共用同一 Waline 库，评论只记录文章路径、不含域名；
 * 不在本站文章路径集合里的评论即视为来自对站（两站文章路径基本互不重叠）。
 */
function resolveCommentSite(
  path: string,
  role: WidgetsProps["recentCommentsSiteRole"],
  ownPaths: Set<string>,
): "main" | "archive" {
  const isOwn = ownPaths.has(decodePath(path));
  const isMainSite = role !== "archive";
  return isOwn === isMainSite ? "main" : "archive";
}

function mapRecentComment(comment: unknown): RecentCommentItem {
  const value = isRecord(comment) ? comment : {};

  const nick = safeString(value.nick ?? t("footer.commentAnonymous"));
  const time = formatDateTime(value.insertedAt ?? value.time ?? value.updatedAt);
  const text = toPlainText(value.comment ?? value.text ?? "");
  const basePath = normalizePath(safeString(value.url ?? value.path ?? "/"));
  const id = safeString(value.objectId ?? value.id ?? value._id ?? "");

  return {
    nick,
    time,
    text,
    href: id ? `${basePath}#waline-comment-${id}` : basePath,
    path: basePath,
  };
}

function truncateText(text: string, maxLength: number = 50): string {
  if (text.length > maxLength) {
    return `${text.substring(0, maxLength)}...`;
  }
  return text;
}

function Widgets(props: WidgetsProps) {
  const [randomPosts, setRandomPosts] = createSignal<Post[]>([]);
  const [recentComments, setRecentComments] =
    createSignal<RecentCommentItemWithSite[]>([]);
  const [loadFailed, setLoadFailed] = createSignal(false);

  const hasWaline = () => Boolean(props.walineServerURL);

  onMount(() => {
    let destroyRecentComments: (() => void) | undefined;

    // 随机文章
    if (props.enableRandomPosts !== false && (props.posts?.length ?? 0) > 0) {
      setRandomPosts(shuffle([...props.posts!]).slice(0, 10));
    }

    // 从 Waline 拉取近期评论（用本站文章路径集合判定每条评论的来源）
    if (props.enableRecentComments !== false && hasWaline()) {
      const walineServerURL = props.walineServerURL ?? "";
      const ownPostPaths = createOwnPostPaths(props.posts);
      const loadRecentComments = async () => {
        const { RecentComments } = await import("@waline/client");
        try {
          const result = await RecentComments({
            serverURL: walineServerURL,
            count: props.recentCommentsLimit ?? 6,
          });
          destroyRecentComments = result.destroy;
          // Waline 的 TS 定义缺少 data 属性
          const data =
            isRecord(result.comments) && Array.isArray(result.comments.data)
              ? result.comments.data
              : [];
          setRecentComments(
            data.map((comment) => {
              const item = mapRecentComment(comment);
              return {
                ...item,
                site: resolveCommentSite(
                  item.path,
                  props.recentCommentsSiteRole,
                  ownPostPaths,
                ),
              };
            }),
          );
        } catch {
          setLoadFailed(true);
          setRecentComments([]);
        }
      };

      void loadRecentComments();
    }

    onCleanup(() => {
      destroyRecentComments?.();
    });
  });

  return (
    <aside class="widgets bg-body-bg-shadow px-4 flex gap-4 justify-around z-1">
      {props.enableRandomPosts !== false && randomPosts().length > 0 && (
        <div class="rpost px-4 py-4 w-1/2">
          <h2 class="text-base color-grey-5 font-semibold m-0 mb-4">{t("footer.randomPosts")}</h2>
          <ul class="post-list m-0 p-0 list-none color-grey-5">
            <For each={randomPosts()}>
              {(post) => (
                <li class="item border-grey-4 pb-2 pl-8 border-b border-dashed relative">
                  <a
                    href={toPostHref(post.slug || post.id)}
                    class="hover:text-color-link text-inherit no-underline flex flex-col transition-colors"
                  >
                    <span class="widget-title text-sm font-semibold m-0 max-h-6">
                      {post.data.title}
                    </span>
                    <span class="text-grey-5 text-xs mt-1 max-h-8">
                      {truncateText(post.data.description || "")}
                    </span>
                  </a>
                </li>
              )}
            </For>
          </ul>
        </div>
      )}

      {props.enableRecentComments !== false && hasWaline() && (
        <div class="rpost px-4 py-4 w-1/2">
          <h2 class="text-base font-semibold m-0 mb-4">{t("footer.recentComments")}</h2>
          <ul id="recent-comment" class="post-list m-0 p-0 list-none">
            {recentComments().length > 0 ? (
              <For each={recentComments()}>
                {(comment) => (
                  <li class="item border-grey-4 pb-2 pl-8 border-b border-dashed relative">
                    <a
                      href={comment.href}
                      class="hover:text-color-link text-inherit no-underline flex flex-col transition-colors"
                    >
                      <span class="widget-title text-sm font-semibold m-0 max-h-6">
                        <span class={`site-tag site-tag-${comment.site}`}>
                          {comment.site === "main"
                            ? t("footer.commentSiteMain")
                            : t("footer.commentSiteArchive")}
                        </span>
                        {comment.nick} @ {comment.time}
                      </span>
                      <span class="text-grey-5 text-xs mt-1 max-h-8">
                        {truncateText(comment.text)}
                      </span>
                    </a>
                  </li>
                )}
              </For>
            ) : loadFailed() ? (
              <li class="item no-counter text-grey-5 py-4 text-center">
                {t("footer.recentCommentsLoadFailed")}
              </li>
            ) : (
              <li class="item no-counter text-grey-5 py-4 text-center">
                {t("footer.noRecentComments")}
              </li>
            )}
          </ul>
        </div>
      )}
    </aside>
  );
}

export default Widgets;
