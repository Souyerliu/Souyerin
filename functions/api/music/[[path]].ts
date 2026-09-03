const METING_API_ORIGIN = "https://meting.api.zkz098.cn";
const NETEASE_API_ORIGIN = "https://music.163.com";
const UPSTREAM_USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36";

const ALLOWED_PATH = /^\/v1\/(?:playlists|songs|albums|artists)\/[A-Za-z0-9_-]+(?:\/lyric)?\/?$/;
const NETEASE_PLAYLIST_PATH = /^\/v1\/playlists\/([A-Za-z0-9_-]+)\/?$/;
const FALLBACK_SONG_LIMIT = 1000;
const FALLBACK_BATCH_SIZE = 250;

interface PagesFunctionContext {
  request: Request;
}

interface NeteaseSong {
  id?: number | string;
  name?: string;
  ar?: Array<{ name?: string }>;
  al?: { picUrl?: string };
}

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const record: Record<string, unknown> = {};
  for (const key of Object.keys(value)) record[key] = Reflect.get(value, key);
  return record;
};

const getString = (value: unknown) => (typeof value === "string" ? value : "");

const getSongId = (value: unknown) => {
  if (typeof value === "number" && Number.isSafeInteger(value)) return String(value);
  if (typeof value === "string" && /^[0-9]+$/.test(value)) return value;
  return "";
};

const toMetingSong = (song: NeteaseSong) => {
  const id = getSongId(song.id);
  if (!id) return null;

  return {
    id,
    name: getString(song.name),
    artist: Array.isArray(song.ar)
      ? song.ar
          .map((artist) => getString(artist?.name))
          .filter(Boolean)
          .join(" / ")
      : "",
    pic_url: getString(song.al?.picUrl),
  };
};

const getUpstreamSongs = (body: string) => {
  try {
    const parsed = asRecord(JSON.parse(body));
    const data = asRecord(parsed?.data);
    return Array.isArray(data?.songs) ? data.songs : null;
  } catch {
    return null;
  }
};

/**
 * meting 上游偶尔会对网易云歌单返回空 songs，但网易云 v6 接口仍能返回歌单索引。
 * 这里只在上游明确返回空歌单时回退，避免正常请求增加额外开销。
 */
const readNeteaseSong = (value: unknown): NeteaseSong | null => {
  const song = asRecord(value);
  if (!song) return null;

  const id = song.id;
  if (
    !(typeof id === "number" && Number.isSafeInteger(id)) &&
    !(typeof id === "string" && /^[0-9]+$/.test(id))
  ) {
    return null;
  }

  const artists = Array.isArray(song.ar)
    ? song.ar.flatMap((artist) => {
        const name = getString(asRecord(artist)?.name);
        return name ? [{ name }] : [];
      })
    : [];
  const picUrl = getString(asRecord(song.al)?.picUrl);

  return {
    id,
    name: getString(song.name),
    ar: artists,
    al: picUrl ? { picUrl } : undefined,
  };
};

const readNeteaseSongs = (value: unknown) =>
  Array.isArray(value)
    ? value.flatMap((song) => {
        const parsedSong = readNeteaseSong(song);
        return parsedSong ? [parsedSong] : [];
      })
    : [];

const fetchNeteasePlaylistFallback = async (playlistId: string) => {
  const headers = new Headers({
    Accept: "application/json",
    Referer: `${NETEASE_API_ORIGIN}/`,
    "User-Agent": UPSTREAM_USER_AGENT,
  });

  const detailResponse = await fetch(
    `${NETEASE_API_ORIGIN}/api/v6/playlist/detail?id=${encodeURIComponent(playlistId)}&n=1000&s=8`,
    { headers, signal: AbortSignal.timeout(10_000) },
  );
  if (!detailResponse.ok) return null;

  const detailPayload = asRecord(await detailResponse.json());
  const playlist = asRecord(detailPayload?.playlist);
  if (detailPayload?.code !== 200 || !playlist) return null;

  const ids = Array.isArray(playlist.trackIds)
    ? playlist.trackIds
        .map((track) => getSongId(asRecord(track)?.id))
        .filter(Boolean)
        .slice(0, FALLBACK_SONG_LIMIT)
    : [];
  if (ids.length === 0) return null;

  const songsById = new Map<string, NeteaseSong>();
  for (const song of readNeteaseSongs(playlist.tracks)) {
    const id = getSongId(song.id);
    if (id) songsById.set(id, song);
  }

  const missingIds = ids.filter((id) => !songsById.has(id));
  const batches = [];
  for (let index = 0; index < missingIds.length; index += FALLBACK_BATCH_SIZE) {
    batches.push(missingIds.slice(index, index + FALLBACK_BATCH_SIZE));
  }

  const batchResults = await Promise.all(
    batches.map(async (batch) => {
      const songsResponse = await fetch(
        `${NETEASE_API_ORIGIN}/api/v3/song/detail?c=${encodeURIComponent(JSON.stringify(batch.map((id) => ({ id: Number(id) }))))}`,
        { headers, signal: AbortSignal.timeout(10_000) },
      );
      if (!songsResponse.ok) return null;

      const songsPayload = asRecord(await songsResponse.json());
      if (songsPayload?.code !== 200) return null;
      return readNeteaseSongs(songsPayload.songs);
    }),
  );
  if (batchResults.some((result) => result === null)) return null;

  for (const songs of batchResults) {
    for (const song of songs ?? []) {
      const id = getSongId(song.id);
      if (id) songsById.set(id, song);
    }
  }

  const songs = ids
    .map((id) => songsById.get(id))
    .filter((song): song is NeteaseSong => Boolean(song));
  const mappedSongs = songs
    .map(toMetingSong)
    .filter((song): song is NonNullable<ReturnType<typeof toMetingSong>> => Boolean(song));
  if (mappedSongs.length === 0) return null;

  return {
    code: 0,
    message: "ok",
    data: {
      id: playlistId,
      platform: "netease",
      songs: mappedSongs,
    },
  };
};

const getNeteasePlaylistId = (upstreamPath: string, requestURL: URL) => {
  const playlistMatch = NETEASE_PLAYLIST_PATH.exec(upstreamPath);
  const platform = requestURL.searchParams.get("platform");

  return playlistMatch && (platform === "netease" || !platform) ? playlistMatch[1] : null;
};

const createFallbackResponse = async (playlistId: string | null) => {
  if (!playlistId) return null;

  try {
    const fallback = await fetchNeteasePlaylistFallback(playlistId);
    if (!fallback) return null;

    return Response.json(fallback, {
      headers: {
        "Cache-Control": "public, max-age=300",
        "X-Music-Source": "netease-v6-fallback",
      },
    });
  } catch {
    return null;
  }
};

/** 代理播放器使用的歌单和歌词接口，避免第三方 API 按站点来源限制跨域。 */
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
  const playlistId = getNeteasePlaylistId(upstreamPath, requestURL);

  let upstreamResponse: Response;
  try {
    const upstreamHeaders = new Headers({
      Accept: "application/json",
      "User-Agent": UPSTREAM_USER_AGENT,
    });
    const acceptLanguage = request.headers.get("Accept-Language");
    if (acceptLanguage) upstreamHeaders.set("Accept-Language", acceptLanguage);

    upstreamResponse = await fetch(upstreamURL, {
      method: request.method,
      headers: upstreamHeaders,
      signal: AbortSignal.timeout(10_000),
    });
  } catch {
    const fallbackResponse =
      request.method === "GET" ? await createFallbackResponse(playlistId) : null;
    if (fallbackResponse) return fallbackResponse;

    return Response.json({ error: "Music service is temporarily unavailable" }, { status: 502 });
  }

  if (!upstreamResponse.ok && request.method === "GET") {
    const fallbackResponse = await createFallbackResponse(playlistId);
    if (fallbackResponse) return fallbackResponse;
  }

  const headers = new Headers();
  const contentType = upstreamResponse.headers.get("Content-Type");
  if (contentType) headers.set("Content-Type", contentType);

  let responseBody: BodyInit | null = request.method === "HEAD" ? null : upstreamResponse.body;
  let shouldCache = upstreamResponse.ok && responseBody !== null;

  if (upstreamResponse.ok && request.method !== "HEAD") {
    const body = await upstreamResponse.text();
    responseBody = body;

    if (playlistId && getUpstreamSongs(body)?.length === 0) {
      const fallbackResponse = await createFallbackResponse(playlistId);
      if (fallbackResponse) return fallbackResponse;

      // 不缓存空歌单，让播放器刷新后可以重新请求上游。
      shouldCache = false;
    }
  }

  if (shouldCache) {
    headers.set("Cache-Control", "public, max-age=300");
  } else if (responseBody !== null) {
    headers.set("Cache-Control", "no-store");
  }

  return new Response(responseBody, {
    status: upstreamResponse.status,
    headers,
  });
};
