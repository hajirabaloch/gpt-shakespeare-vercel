import { useState } from "react";

export default function App() {
  const [prompt, setPrompt] = useState("ROMEO:");
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(200);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function onGenerate() {
    setLoading(true);
    setError("");
    try {
      const r = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, temperature: Number(temperature), maxTokens: Number(maxTokens) }),
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(d.error || `Request failed (${r.status})`);
      setResult({ prompt, ...d });
    } catch (e) {
      setError(e.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function onClear() {
    setPrompt("");
    setResult(null);
    setError("");
  }

  return (
    <main className="wrap">
      <h1>Tiny Shakespeare GPT</h1>
      <p className="sub">
        A decoder-only Transformer trained from scratch to predict the next character. It runs in native
        JavaScript on this server, with no AI APIs.
      </p>

      <label htmlFor="prompt">Prompt</label>
      <textarea id="prompt" rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)}
        placeholder="Type a few words, e.g. ROMEO:" />

      <div className="controls">
        <div>
          <label htmlFor="temp">Temperature: {Number(temperature).toFixed(2)}</label>
          <input id="temp" type="range" min="0.1" max="2" step="0.05" value={temperature}
            onChange={(e) => setTemperature(e.target.value)} />
        </div>
        <div>
          <label htmlFor="max">Max new characters: {maxTokens}</label>
          <input id="max" type="range" min="10" max="300" step="10" value={maxTokens}
            onChange={(e) => setMaxTokens(e.target.value)} />
        </div>
      </div>

      <div className="buttons">
        <button className="primary" onClick={onGenerate} disabled={loading}>
          {loading ? "Generating..." : "Generate"}
        </button>
        <button onClick={onClear} disabled={loading}>Clear</button>
      </div>

      {loading && <div className="spinner" role="status" aria-label="Loading" />}
      {error && <div className="error" role="alert">{error}</div>}

      {result && (
        <section className="output">
          <pre><span className="prompt">{result.prompt}</span>{result.generated}</pre>
          <small>
            {result.generated.length} characters in {result.ms} ms
            {result.ignoredChars ? ` | ignored characters not in the vocabulary: ${JSON.stringify(result.ignoredChars)}` : ""}
          </small>
        </section>
      )}
    </main>
  );
}
