// Optional Basic Auth for the whole site. The API key only keeps people away from the API
// directly: without this, anyone who can open the site can use it through the backend route.
import { NextResponse, type NextRequest } from "next/server";

const PASSWORD = process.env.WEB_PASSWORD;

function authorized(request: NextRequest): boolean {
  const [scheme, encoded] = (request.headers.get("authorization") ?? "").split(" ");
  if (scheme?.toLowerCase() !== "basic" || !encoded) return false;
  try {
    const decoded = atob(encoded);
    return decoded.slice(decoded.indexOf(":") + 1) === PASSWORD;
  } catch {
    return false; // not valid base64
  }
}

export function proxy(request: NextRequest) {
  if (!PASSWORD || authorized(request)) return NextResponse.next();
  return new NextResponse("Authentication required", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="DocPilot"' },
  });
}

export const config = { matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"] };
