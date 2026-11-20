// Calls the real API handler with fake req/res objects (no server needed).
import handler from "../api/generate.js";

function call(method, body) {
  return new Promise((resolve) => {
    const res = {
      code: 200, headers: {},
      setHeader(k, v) { this.headers[k] = v; },
      status(c) { this.code = c; return this; },
      json(o) { resolve({ code: this.code, body: o }); },
    };
    handler({ method, body }, res);
  });
}

let ok = true;
const check = (name, cond) => { console.log(cond ? "PASS" : "FAIL", name); if (!cond) ok = false; };

const good = await call("POST", { prompt: "ROMEO:", temperature: 0.8, maxTokens: 50 });
check("valid request returns 200", good.code === 200);
check("returns 50 generated characters", good.body.generated && good.body.generated.length === 50);
console.log("sample:", JSON.stringify(good.body.generated));
check("rejects bad temperature", (await call("POST", { prompt: "a", temperature: 9, maxTokens: 5 })).code === 400);
check("rejects bad maxTokens", (await call("POST", { prompt: "a", temperature: 1, maxTokens: 100000 })).code === 400);
check("rejects non-string prompt", (await call("POST", { prompt: 5 })).code === 400);
check("rejects missing body", (await call("POST", undefined)).code === 400);
check("GET health works", (await call("GET")).code === 200);
check("PUT is 405", (await call("PUT", {})).code === 405);
process.exit(ok ? 0 : 1);
