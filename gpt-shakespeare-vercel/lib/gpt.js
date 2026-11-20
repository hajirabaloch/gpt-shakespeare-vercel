import fs from "node:fs";
import path from "node:path";

// ---------- loading ----------
export function loadModel(dir) {
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, "model_manifest.json"), "utf8"));
  const tok = JSON.parse(fs.readFileSync(path.join(dir, "tokenizer.json"), "utf8"));
  const raw = fs.readFileSync(path.join(dir, "weights.bin"));
  const all = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const W = {};
  for (const t of manifest.tensors) W[t.name] = all.subarray(t.offset, t.offset + t.length);
  return { cfg: manifest.config, W, tok, meta: manifest.metadata };
}

// ---------- tokenizer ----------
export function encode(tok, text) {
  const ids = [];
  const ignored = [];
  for (const ch of text) {
    if (Object.prototype.hasOwnProperty.call(tok.stoi, ch)) ids.push(tok.stoi[ch]);
    else ignored.push(ch);
  }
  return { ids, ignored };
}
export function decode(tok, ids) {
  return ids.map((i) => tok.itos[i]).join("");
}

// ---------- math building blocks ----------
function layerNorm(x, w, b, T, C) {
  const out = new Float32Array(T * C);
  for (let t = 0; t < T; t++) {
    const o = t * C;
    let mean = 0;
    for (let c = 0; c < C; c++) mean += x[o + c];
    mean /= C;
    let v = 0;
    for (let c = 0; c < C; c++) { const d = x[o + c] - mean; v += d * d; }
    v /= C; // biased variance, like PyTorch
    const inv = 1 / Math.sqrt(v + 1e-5);
    for (let c = 0; c < C; c++) out[o + c] = (x[o + c] - mean) * inv * w[c] + b[c];
  }
  return out;
}

// y = x W^T + b, with W shaped [nOut, nIn] (PyTorch nn.Linear layout)
function linear(x, T, nIn, w, b, nOut) {
  const out = new Float32Array(T * nOut);
  for (let t = 0; t < T; t++) {
    const xo = t * nIn;
    for (let o = 0; o < nOut; o++) {
      let s = b[o];
      const wo = o * nIn;
      for (let i = 0; i < nIn; i++) s += x[xo + i] * w[wo + i];
      out[t * nOut + o] = s;
    }
  }
  return out;
}

const GELU_K = Math.sqrt(2 / Math.PI);
function geluTanh(x) {
  return 0.5 * x * (1 + Math.tanh(GELU_K * (x + 0.044715 * x * x * x)));
}

export function softmax(logits, temperature = 1) {
  const n = logits.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
  const p = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) { p[i] = Math.exp((logits[i] - max) / temperature); sum += p[i]; }
  for (let i = 0; i < n; i++) p[i] /= sum;
  return p;
}

// ---------- the transformer forward pass: returns next-token logits for the LAST position ----------
export function forward(m, ids) {
  const { n_embd: C, n_head: H, n_layer: L, ff_dim: FF, vocab_size: V, block_size } = m.cfg;
  const W = m.W;
  const T = ids.length;
  if (T < 1 || T > block_size) throw new Error(`sequence length must be 1..${block_size}`);
  const hs = C / H;
  const scale = 1 / Math.sqrt(hs);

  // token + positional embeddings
  let x = new Float32Array(T * C);
  for (let t = 0; t < T; t++) {
    const id = ids[t];
    for (let c = 0; c < C; c++) {
      x[t * C + c] = W["tok_emb.weight"][id * C + c] + W["pos_emb.weight"][t * C + c];
    }
  }

  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`;
    // --- causal multi-head self-attention ---
    const h = layerNorm(x, W[p + "ln1.weight"], W[p + "ln1.bias"], T, C);
    const qkv = linear(h, T, C, W[p + "attn.qkv.weight"], W[p + "attn.qkv.bias"], 3 * C);
    const y = new Float32Array(T * C);
    const att = new Float64Array(T);
    for (let head = 0; head < H; head++) {
      const ho = head * hs;
      for (let t = 0; t < T; t++) {
        let max = -Infinity;
        for (let s = 0; s <= t; s++) { // causal mask: only s <= t
          let dot = 0;
          for (let d = 0; d < hs; d++) dot += qkv[t * 3 * C + ho + d] * qkv[s * 3 * C + C + ho + d];
          att[s] = dot * scale;
          if (att[s] > max) max = att[s];
        }
        let sum = 0;
        for (let s = 0; s <= t; s++) { att[s] = Math.exp(att[s] - max); sum += att[s]; }
        for (let d = 0; d < hs; d++) {
          let acc = 0;
          for (let s = 0; s <= t; s++) acc += (att[s] / sum) * qkv[s * 3 * C + 2 * C + ho + d];
          y[t * C + ho + d] = acc;
        }
      }
    }
    const a = linear(y, T, C, W[p + "attn.proj.weight"], W[p + "attn.proj.bias"], C);
    for (let i = 0; i < x.length; i++) x[i] += a[i]; // residual 1

    // --- feed-forward ---
    const h2 = layerNorm(x, W[p + "ln2.weight"], W[p + "ln2.bias"], T, C);
    const f = linear(h2, T, C, W[p + "ff.0.weight"], W[p + "ff.0.bias"], FF);
    for (let i = 0; i < f.length; i++) f[i] = geluTanh(f[i]);
    const g = linear(f, T, FF, W[p + "ff.2.weight"], W[p + "ff.2.bias"], C);
    for (let i = 0; i < x.length; i++) x[i] += g[i]; // residual 2
  }

  const xf = layerNorm(x, W["ln_f.weight"], W["ln_f.bias"], T, C);
  const last = xf.subarray((T - 1) * C, T * C);
  return linear(last, 1, C, W["lm_head.weight"], W["lm_head.bias"], V);
}

// ---------- autoregressive generation ----------
export function sampleIndex(probs, rand = Math.random) {
  let r = rand();
  for (let i = 0; i < probs.length; i++) { r -= probs[i]; if (r <= 0) return i; }
  return probs.length - 1;
}

export function generate(m, promptIds, maxNewTokens, temperature = 0.8, rand = Math.random) {
  const ids = promptIds.length ? [...promptIds] : [m.tok.stoi["\n"]];
  const out = [];
  for (let n = 0; n < maxNewTokens; n++) {
    const ctx = ids.slice(-m.cfg.block_size);
    const probs = softmax(forward(m, ctx), temperature);
    const next = sampleIndex(probs, rand);
    ids.push(next);
    out.push(next);
  }
  return decode(m.tok, out);
}
