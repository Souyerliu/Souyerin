const METING_API_ORIGIN = "https://meting.api.zkz098.cn";

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
      Accept: request.headers.get("Accept") || "application/json",
      // 上游 API 的 Cloudflare 防护会拦截无 User-Agent 的边缘请求。
      "User-Agent": request.headers.get("User-Agent") || "Mozilla/5.0 (compatible; MusicProxy/1.0)",
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
