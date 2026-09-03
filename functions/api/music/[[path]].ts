const METING_API_ORIGIN = "https://meting.api.zkz098.cn";
const UPSTREAM_USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36";

const ALLOWED_PATH = /^\/v1\/(?:playlists|songs|albums|artists)\/[A-Za-z0-9_-]+(?:\/lyric)?\/?$/;

interface PagesFunctionContext {
  request: Request;
}

/**
 * 代理 Nyx Player 使用的歌单和歌词接口，避免第三方 API 按站点来源限制跨域。
 *
 * 这里只放行播放器实际需要的只读路径，避免把 Pages Function 变成开放代理。
 */
export const onRequest = async ({ request }: PagesFunctionContext) => {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return new Response("Method Not Allowed", {
      status: 405,
      headers: { Allow: "GET, HEAD" },
    });
  }

  const requestURL = new URL(request.url);
  const upstreamPath = requestURL.pathname.replace(/^\/api\/music(?=\/|$)/, "");

  if (!ALLOWED_PATH.test(upstreamPath)) {
    return new Response("Not Found", { status: 404 });
  }

  const upstreamURL = new URL(upstreamPath, METING_API_ORIGIN);
  upstreamURL.search = requestURL.search;

  let upstreamResponse: Response;
  try {
    const upstreamHeaders = new Headers({
      Accept: "application/json",
      // 上游 API 会拦截 curl 等非浏览器 User-Agent，统一使用浏览器标识。
      "User-Agent": UPSTREAM_USER_AGENT,
    });
    const acceptLanguage = request.headers.get("Accept-Language");
    if (acceptLanguage) {
      upstreamHeaders.set("Accept-Language", acceptLanguage);
    }

    upstreamResponse = await fetch(upstreamURL, {
      method: request.method,
      headers: upstreamHeaders,
      signal: AbortSignal.timeout(10_000),
    });
  } catch {
    return Response.json({ error: "Music service is temporarily unavailable" }, { status: 502 });
  }

  const headers = new Headers();
  const contentType = upstreamResponse.headers.get("Content-Type");
  if (contentType) {
    headers.set("Content-Type", contentType);
  }

  // 歌单和歌词均为公开只读数据，可在边缘短暂缓存，减轻上游压力。
  if (upstreamResponse.ok) {
    headers.set("Cache-Control", "public, max-age=300");
  }

  return new Response(request.method === "HEAD" ? null : upstreamResponse.body, {
    status: upstreamResponse.status,
    headers,
  });
};
