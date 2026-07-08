// Simple static file server for GLB viewing
import { serve } from "https://deno.land/std@0.224.0/http/server.ts";

const PORT = 8766;
const ROOT = ".";

function mimeType(path: string): string {
  if (path.endsWith(".html")) return "text/html; charset=utf-8";
  if (path.endsWith(".glb")) return "model/gltf-binary";
  if (path.endsWith(".js")) return "application/javascript";
  if (path.endsWith(".json")) return "application/json";
  if (path.endsWith(".png")) return "image/png";
  if (path.endsWith(".jpg") || path.endsWith(".jpeg")) return "image/jpeg";
  return "application/octet-stream";
}

serve((req) => {
  const url = new URL(req.url);
  let path = url.pathname;
  if (path === "/") path = "/glb-viewer.html";
  const filePath = ROOT + path;
  try {
    const data = Deno.readFileSync(filePath);
    return new Response(data, { headers: { "Content-Type": mimeType(filePath) } });
  } catch {
    return new Response("Not found: " + path, { status: 404 });
  }
}, { port: PORT });

console.log(`Server running on http://localhost:${PORT}/`);
