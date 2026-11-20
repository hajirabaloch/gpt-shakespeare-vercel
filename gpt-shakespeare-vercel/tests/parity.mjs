// Compares JavaScript output against PyTorch reference values (same weights, same inputs).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadModel, forward, softmax } from "../lib/gpt.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const vec = JSON.parse(fs.readFileSync(path.join(here, "test_vectors.json"), "utf8"));
const m = loadModel(path.join(here, "..", "model"));

let maxLogit = 0, maxProb = 0, argmaxOk = true;
const rows = [];
for (const c of vec.cases) {
  const logits = forward(m, c.ids);
  const probs = softmax(logits, 1);
  let jsArg = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[jsArg]) jsArg = i;
  let eL = 0, eP = 0;
  for (let i = 0; i < logits.length; i++) {
    eL = Math.max(eL, Math.abs(logits[i] - c.logits[i]));
    eP = Math.max(eP, Math.abs(probs[i] - c.probs[i]));
  }
  maxLogit = Math.max(maxLogit, eL);
  maxProb = Math.max(maxProb, eP);
  if (jsArg !== c.argmax) argmaxOk = false;
  rows.push({ name: c.name, length: c.ids.length, pyArgmax: c.argmax, jsArgmax: jsArg, maxLogitErr: eL, maxProbErr: eP });
}
const pass = argmaxOk && maxLogit < 1e-3 && maxProb < 1e-4;
const report = { cases: rows, maxAbsLogitError: maxLogit, maxAbsProbError: maxProb, argmaxAllMatch: argmaxOk, pass };
fs.writeFileSync(path.join(here, "parity_report.json"), JSON.stringify(report, null, 2));
console.table(rows);
console.log(`max |logit error| = ${maxLogit.toExponential(3)}`);
console.log(`max |prob error|  = ${maxProb.toExponential(3)}`);
console.log(`predicted tokens all match: ${argmaxOk}`);
console.log(pass ? "PARITY PASSED" : "PARITY FAILED");
process.exit(pass ? 0 : 1);
