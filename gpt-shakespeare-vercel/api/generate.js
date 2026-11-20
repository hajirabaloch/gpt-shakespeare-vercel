import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadModel, encode, generate } from "../lib/gpt.js";

// Loaded once per serverless instance (warm requests reuse it).
let model = null;
function getModel() {
  if (model) return model;
  const here = path.dirname(fileURLToPath(import.meta.url));
  const candidates = [path.join(process.cwd(), "model"), path.join(here, "..", "model")];
  const dir = candidates.find((d) => fs.existsSync(path.join(d, "weights.bin")));
  if (!dir) throw new Error("model files not found");
  model = loadModel(dir);
  return model;
}

export default function handler(req, res) {
  try {
    if (req.method === "GET") {
      const m = getModel();
      return res.status(200).json({ status: "ok", config: m.cfg, metadata: m.meta });
    }
    if (req.method !== "POST") {
      res.setHeader("Allow", "GET, POST");
      return res.status(405).json({ error: "Use POST." });
    }
    const body = req.body;
    if (!body || typeof body !== "object") return res.status(400).json({ error: "Send a JSON body." });

    const prompt = body.prompt === undefined ? "" : body.prompt;
    const temperature = body.temperature === undefined ? 0.8 : body.temperature;
    const maxTokens = body.maxTokens === undefined ? 200 : body.maxTokens;

    if (typeof prompt !== "string" || prompt.length > 2000)
      return res.status(400).json({ error: "prompt must be a string up to 2000 characters." });
    if (typeof temperature !== "number" || !(temperature >= 0.1 && temperature <= 2))
      return res.status(400).json({ error: "temperature must be a number between 0.1 and 2." });
    if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 300)
      return res.status(400).json({ error: "maxTokens must be an integer between 1 and 300." });

    const m = getModel();
    const { ids, ignored } = encode(m.tok, prompt);
    const t0 = Date.now();
    const generated = generate(m, ids, maxTokens, temperature);
    return res.status(200).json({
      generated,
      ignoredChars: [...new Set(ignored)].join(""),
      temperature,
      maxTokens,
      ms: Date.now() - t0,
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "Internal error while generating text." });
  }
}
