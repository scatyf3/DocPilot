// Server-side proxy to the FastAPI backend. The API key stays on the server; browsers only ever
// talk to this route. Limited to the paths the UI uses, so it can't reach anything else.
const API_URL = process.env.API_URL ?? "http://localhost:8000";
const API_KEY = process.env.API_KEY;
// Next.js keeps an x-forwarded-for sent by the client, so it only identifies clients behind a
// platform that overwrites it (Vercel does). Otherwise all traffic shares one rate-limit bucket,
// which fails safe: strict limits rather than limits anyone can dodge with a made-up header.
const TRUST_FORWARDED_FOR = process.env.TRUST_FORWARDED_FOR === "true";
const ALLOWED = /^(documents(\/\d+)?|chat|traces\/[0-9a-f-]+|conversations\/[^/]+)$/;

// Connection-specific, or describing the body as received rather than as we re-send it.
// x-client-ip is ours to set: a browser sending its own would dodge the rate limits.
const DROP_REQUEST = [
  "host", "connection", "content-length", "accept-encoding", "authorization", "cookie", "x-client-ip",
  "transfer-encoding", "keep-alive", "upgrade", "te", "trailer", "proxy-authorization",
  // uvicorn believes these from a local peer, so passing them on would let clients pick their own IP.
  "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "x-forwarded-port", "x-real-ip", "forwarded",
];
const DROP_RESPONSE = ["connection", "content-length", "content-encoding", "transfer-encoding"];

async function forward(request: Request, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  if (!ALLOWED.test(path.join("/"))) return Response.json({ detail: "not found" }, { status: 404 });

  const target = path.map(encodeURIComponent).join("/");
  const headers = new Headers(request.headers);
  DROP_REQUEST.forEach((h) => headers.delete(h));
  if (API_KEY) headers.set("authorization", `Bearer ${API_KEY}`);
  if (TRUST_FORWARDED_FOR) {
    // Every request reaches the API from this server, so per-client limits need the real client.
    const client = request.headers.get("x-forwarded-for")?.split(",")[0].trim();
    if (client) headers.set("x-client-ip", client);
  }

  const hasBody = request.method !== "GET" && request.method !== "HEAD";
  let upstream: Response;
  try {
    upstream = await fetch(`${API_URL}/${target}${new URL(request.url).search}`, {
      method: request.method,
      headers,
      body: hasBody ? request.body : undefined,
      signal: request.signal, // stopping a chat in the UI also stops the agent
      ...(hasBody ? { duplex: "half" } : {}),
    } as RequestInit);
  } catch {
    return Response.json({ detail: "the API is unavailable" }, { status: 502 });
  }

  const responseHeaders = new Headers(upstream.headers);
  DROP_RESPONSE.forEach((h) => responseHeaders.delete(h));
  return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
}

export { forward as GET, forward as POST, forward as DELETE };
