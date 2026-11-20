# PROJECT_CONTEXT

Goal: university project. Build a decoder-only Transformer (GPT), train it from scratch, predict the next token, and deploy a web demo on Vercel Hobby.

## Decisions
- Data: Tiny Shakespeare (1,115,394 characters), character-level tokenizer, 90/10 train/val split.
- Model: context 128, embedding 128, 4 heads, 2 blocks, feed-forward 512, dropout 0.1, pre-LayerNorm, GELU (tanh), untied output head. 429,889 parameters.
- Training: AdamW, lr 2e-3 with warmup and cosine decay, batch 64, grad clip 1.0, seed 1337, 5000 steps. Best checkpoint chosen by validation loss.
- Export: float32 little-endian `weights.bin` + `model_manifest.json` (tensor names, shapes, offsets, config, metadata).
- Inference: plain JavaScript (`lib/gpt.js`), no ML libraries. The model is loaded once per serverless instance.
- Verification: PyTorch (rebuilt from exported files) vs JavaScript on 6 inputs. Max abs logit error 9.54e-06, all predicted tokens matched.
- Deployment: Vercel Hobby, Vite static frontend + `api/generate.js`; `vercel.json` includes `model/**` in the function bundle.

## Status
Trained, exported and verified inside the Colab notebook. API handler tested with a local smoke test. Live Vercel deployment is done by the repository owner and is not claimed here.

## Limits
API caps: 300 new characters per request, temperature 0.1 to 2, prompt up to 2000 characters. Characters not in the vocabulary are ignored. Each generated character recomputes the full context (no KV cache), which is fine for this small model.
