// Local API server for development: `npm run api` (Vite proxies /api to it).
import http from "node:http";
import handler from "./api/generate.js";

http.createServer((req, res) => {
  let data = "";
  req.on("data", (c) => (data += c));
  req.on("end", () => {
    try { req.body = data ? JSON.parse(data) : {}; } catch { req.body = undefined; }
    res.status = (c) => { res.statusCode = c; return res; };
    res.json = (o) => { res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify(o)); };
    handler(req, res);
  });
}).listen(3001, () => console.log("API on http://localhost:3001/api/generate"));
