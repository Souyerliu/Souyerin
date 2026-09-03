import { afterEach, describe, expect, it, vi } from "vitest";

import { onRequest } from "../../functions/api/music/[[path]]";

const playlistURL = "https://example.com/api/music/v1/playlists/2257046115?platform=netease";

const jsonResponse = (body: unknown, status = 200) =>
  Response.json(body, {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });

const requestPlaylist = () => onRequest({ request: new Request(playlistURL) });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("网易云音乐代理", () => {
  it("透传包含歌曲的 Meting 响应", async () => {
    const upstream = {
      code: 0,
      data: { songs: [{ id: "1", name: "Meting song" }] },
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(upstream));
    vi.stubGlobal("fetch", fetchMock);

    const response = await requestPlaylist();

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual(upstream);
    expect(response.headers.get("Cache-Control")).toBe("public, max-age=300");
    expect(response.headers.get("X-Music-Source")).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("Meting 返回空歌单时回退到网易云接口", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ code: 0, data: { songs: [] } }))
      .mockResolvedValueOnce(
        jsonResponse({
          code: 200,
          playlist: {
            trackIds: [{ id: 42 }],
            tracks: [],
          },
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          code: 200,
          songs: [
            {
              id: 42,
              name: "Fallback song",
              ar: [{ name: "Artist" }],
              al: { picUrl: "https://example.com/cover.jpg" },
            },
          ],
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const response = await requestPlaylist();
    const payload = await response.json();

    expect(response.status).toBe(200);
    expect(response.headers.get("X-Music-Source")).toBe("netease-v6-fallback");
    expect(payload.data.songs).toEqual([
      {
        id: "42",
        name: "Fallback song",
        artist: "Artist",
        pic_url: "https://example.com/cover.jpg",
      },
    ]);
  });

  it("Meting 请求异常时也使用网易云回退", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("upstream unavailable"))
      .mockResolvedValueOnce(
        jsonResponse({
          code: 200,
          playlist: {
            trackIds: [{ id: 7 }],
            tracks: [
              {
                id: 7,
                name: "Available song",
                ar: [{ name: "Artist" }],
                al: { picUrl: "https://example.com/cover.jpg" },
              },
            ],
          },
        }),
      )
      .mockResolvedValueOnce(jsonResponse({ code: 200, songs: [] }));
    vi.stubGlobal("fetch", fetchMock);

    const response = await requestPlaylist();
    const payload = await response.json();

    expect(response.status).toBe(200);
    expect(response.headers.get("X-Music-Source")).toBe("netease-v6-fallback");
    expect(payload.data.songs[0].id).toBe("7");
  });

  it("回退失败时不缓存空歌单", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ code: 0, data: { songs: [] } }))
      .mockResolvedValueOnce(jsonResponse({ code: 503 }, 503));
    vi.stubGlobal("fetch", fetchMock);

    const response = await requestPlaylist();

    expect(response.status).toBe(200);
    expect(response.headers.get("Cache-Control")).toBe("no-store");
    expect((await response.json()).data.songs).toEqual([]);
  });
});
