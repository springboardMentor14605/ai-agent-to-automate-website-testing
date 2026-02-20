import { useState } from "react";
import { runTest } from "./api.js";

export default function App() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    setResult(await runTest(url));
  }

  return (
    <main>
      <h1>AI Test Agent</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Enter URL to test"
        />
        <button>Run Test</button>
      </form>

      {result && (
        <div>
          <h2>AI Steps</h2>
          <pre>{JSON.stringify(result.steps, null, 2)}</pre>
          <h2>Result</h2>
          <pre>{JSON.stringify(result.result, null, 2)}</pre>
        </div>
      )}
    </main>
  );
}