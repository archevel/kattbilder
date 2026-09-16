// Minimal static server for ./public (WebGPU + Cache API need http://, not file://).
// Streams files, so multi-GB model weights work.
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "public");
const port = Number(process.env.PORT) || 8080;
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".bin": "application/octet-stream", ".onnx": "application/octet-stream" };

createServer(async (req, res) => {
  let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (p === "/") p = "/index.html";
  const file = path.normalize(path.join(root, p));
  if (!file.startsWith(root)) { res.writeHead(403); return res.end(); }
  try {
    const st = await stat(file);
    if (!st.isFile()) throw new Error("not a file");
    res.writeHead(200, { "content-type": types[path.extname(file)] || "application/octet-stream", "content-length": st.size });
    if (req.method === "HEAD") return res.end();
    createReadStream(file).pipe(res);
  } catch { res.writeHead(404); res.end("not found"); }
}).listen(port, () => console.log(`http://localhost:${port}`));
