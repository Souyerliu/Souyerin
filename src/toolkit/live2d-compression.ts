interface CompressionHost {
  fetch: typeof fetch;
  location: Pick<Location, "href" | "origin">;
  DecompressionStream?: typeof DecompressionStream;
}

/** 只为本站 yuno 核心使用无损压缩，旧浏览器或资源异常时回退原文件。 */
export function installLive2DModelCompression(host: CompressionHost) {
  const Decoder = host.DecompressionStream;
  if (!Decoder) return;
  const originalFetch = host.fetch.bind(host);
  host.fetch = async (input, options) => {
    const sourceRequest = input instanceof Request ? input : undefined;
    const address = input instanceof Request ? input.url : input;
    const url = new URL(address, host.location.href);
    const method = options?.method ?? sourceRequest?.method ?? "GET";
    const headers = new Headers(options?.headers ?? sourceRequest?.headers);
    if (
      url.origin !== host.location.origin ||
      !url.pathname.startsWith("/live2d-models/model/yuno/") ||
      !url.pathname.endsWith(".moc3") ||
      method.toUpperCase() !== "GET" ||
      headers.has("range")
    )
      return originalFetch(input, options);

    url.pathname += ".gz";
    const request = new Request(url, sourceRequest);
    try {
      const compressed = await originalFetch(request, options);
      if (!compressed.ok) throw new Error("压缩模型不可用");
      const bytes = new Uint8Array(await compressed.arrayBuffer());
      // 部分静态托管会设置 Content-Encoding，浏览器已经自动解压，避免再次解压。
      const body =
        bytes[0] === 0x1f && bytes[1] === 0x8b
          ? await new Response(
              new Blob([bytes]).stream().pipeThrough(new Decoder("gzip")),
            ).arrayBuffer()
          : bytes.buffer;
      const magic = new Uint8Array(body, 0, Math.min(body.byteLength, 4));
      if (
        magic.length !== 4 ||
        magic[0] !== 0x4d ||
        magic[1] !== 0x4f ||
        magic[2] !== 0x43 ||
        magic[3] !== 0x33
      ) {
        throw new Error("压缩模型内容无效");
      }
      return new Response(body, { headers: { "Content-Type": "application/octet-stream" } });
    } catch (error) {
      if ((options?.signal ?? request.signal)?.aborted) throw error;
      return originalFetch(input, options);
    }
  };
}
