import { gzipSync } from "node:zlib";
import { describe, expect, it, vi } from "vitest";
import { installLive2DModelCompression } from "./live2d-compression";

const url = "https://blog.test/live2d-models/model/yuno/casual/model.moc3";
const model = new TextEncoder().encode("MOC3 test model data");
function setup(
  response: (input: RequestInfo | URL, options?: RequestInit) => Promise<Response>,
  decoder: typeof DecompressionStream | undefined = DecompressionStream,
) {
  const fetch = vi.fn(response);
  const host = {
    fetch: fetch as typeof globalThis.fetch,
    location: { href: "https://blog.test/about/", origin: "https://blog.test" },
    DecompressionStream: decoder,
  };
  installLive2DModelCompression(host);
  return { host, fetch };
}

describe("Live2D 无损压缩加载", () => {
  it("解压后为原始模型字节，保留请求选项和查询参数", async () => {
    const { host, fetch } = setup(async () => new Response(gzipSync(model)));
    const controller = new AbortController();
    const response = await host.fetch(`${url}?v=2`, {
      signal: controller.signal,
      credentials: "include",
    });
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(model);
    const [request, options] = fetch.mock.calls[0];
    if (!(request instanceof Request) || !options) throw new Error("缺少压缩模型请求");
    expect(request.url).toBe(`${url}.gz?v=2`);
    expect(options.signal).toBe(controller.signal);
    expect(options.credentials).toBe("include");
    expect(fetch).toHaveBeenCalledOnce();
  });

  it("兼容托管平台已自动解压的响应", async () => {
    const { host } = setup(async () => new Response(model));
    expect(new Uint8Array(await (await host.fetch(url)).arrayBuffer())).toEqual(model);
  });

  it.each(["missing", "corrupt", "html", "network"])(
    "压缩资源 %s 时回退原始文件",
    async (failure) => {
      const { host, fetch } = setup(async () => {
        if (fetch.mock.calls.length > 1) return new Response(model);
        if (failure === "network") throw new TypeError("network error");
        if (failure === "missing") return new Response("missing", { status: 404 });
        if (failure === "corrupt") return new Response(new Uint8Array([0x1f, 0x8b, 0]));
        return new Response("<html>404</html>");
      });
      expect(new Uint8Array(await (await host.fetch(url)).arrayBuffer())).toEqual(model);
      expect(fetch).toHaveBeenLastCalledWith(url, undefined);
    },
  );

  it("不修改其他角色、外部请求、非 GET 和范围请求", async () => {
    const { host, fetch } = setup(async () => new Response(model));
    const inputs = [
      url.replace("yuno", "soyo"),
      url.replace("blog.test", "cdn.test"),
      url.replace(".moc3", ".json"),
    ];
    await Promise.all(inputs.map((input) => host.fetch(input)));
    await host.fetch(url, { method: "POST" });
    const request = new Request(url, { headers: { Range: "bytes=0-10" } });
    await host.fetch(request);
    expect(fetch).toHaveBeenCalledTimes(5);
    expect(fetch).toHaveBeenLastCalledWith(request, undefined);
    for (const input of inputs) expect(fetch).toHaveBeenCalledWith(input, undefined);
  });

  it("不支持解压的浏览器直接使用原始加载", async () => {
    const { host, fetch } = setup(async () => new Response(model));
    const unsupported = {
      ...host,
      fetch: fetch as typeof globalThis.fetch,
      DecompressionStream: undefined,
    };
    installLive2DModelCompression(unsupported);
    expect(unsupported.fetch).toBe(fetch);
  });

  it("取消请求不触发回退", async () => {
    const controller = new AbortController();
    const { host, fetch } = setup(async () => {
      controller.abort();
      throw new DOMException("aborted", "AbortError");
    });
    await expect(host.fetch(url, { signal: controller.signal })).rejects.toMatchObject({
      name: "AbortError",
    });
    expect(fetch).toHaveBeenCalledOnce();
  });
});
