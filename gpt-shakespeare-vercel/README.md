# Tiny Shakespeare GPT

A decoder-only Transformer (GPT architecture) **trained from scratch** on Tiny Shakespeare to predict the next character, with a native JavaScript inference engine, a Node.js API and a React + Vite interface, ready for Vercel Hobby.

## Results (measured in the training notebook)
| Item | Value |
|---|---|
| Parameters | 429,889 |
| Training steps | 5000 |
| Final train loss | 1.3089 |
| Final validation loss | 1.5360 |
| Best validation loss | 1.5360 (step 5000) |
| Training time | 119.8 s on cuda |
| Max abs logit error, PyTorch vs JS | 9.54e-06 |
| Max abs probability error, PyTorch vs JS | 1.48e-06 |

Loss curve: `assets/loss_curve.png`. Raw numbers: `assets/metrics.json`.

## Architecture
Character-level tokenizer (vocab 65) → token + positional embeddings → 2 pre-LayerNorm decoder blocks (4-head causal self-attention with Q/K/V projections, GELU feed-forward 128→512→128, residual connections, dropout 0.1) → final LayerNorm → linear head → softmax. Context length 128.

## Repository layout
```
api/generate.js        Vercel serverless function (POST /api/generate)
lib/gpt.js             Transformer inference in plain JavaScript
model/                 weights.bin, model_manifest.json, tokenizer.json (trained weights, about 1.7 MB)
src/, index.html       React + Vite frontend
python/model.py        PyTorch model (training)
python/export_model.py weight export / reload
checkpoints/           best.pt and final.pt (PyTorch)
tests/                 parity.mjs, api_smoke.mjs, test_vectors.json, parity_report.json
assets/                loss curve, metrics, samples
vercel.json            function config (bundles model/**)
```
Add the training notebook to a `notebooks/` folder if your teacher wants it in the repo.

## Run locally
```
npm install
npm run test:parity   # JS vs PyTorch
npm run test:api
npm run api           # terminal 1
npm run dev           # terminal 2 -> http://localhost:5173
```

## Deploy on Vercel (free Hobby)
1. Create a GitHub repository and upload everything in this folder (`node_modules` and `dist` are ignored).
2. Go to vercel.com → Add New → Project → import the repository.
3. Framework Preset: **Vite** (detected from `vercel.json`). No environment variables needed. Click **Deploy**.
4. Open the URL, then open `/api/generate` in the browser: it should return `{"status":"ok", ...}`.

Vercel Hobby limits that matter here (see https://vercel.com/docs/functions/limitations): 250 MB uncompressed function bundle (the model is about 1.7 MB), 2 GB memory and up to 300 s duration on Hobby.

## Why the text is not grammatically perfect
The model has under half a million parameters, sees 128 characters at a time and learns letter by letter from about 1 MB of text. It copies the style of Shakespeare, not meaning.

## Sample output (best model, from the notebook)
```
When should, will Romeo, and they are good touches,
The great of our tears of the forth days,
But love, judge but the consul of this son:
'Twere eyes the accused him with part
And seem them are the wa
```
