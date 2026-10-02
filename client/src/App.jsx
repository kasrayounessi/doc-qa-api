import { useState } from "react";

async function fetchSample(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load sample: ${path}`);
  const blob = await res.blob();
  return new File([blob], path.split("/").pop(), { type: blob.type });
}

function FileRow({ label, file, accept, onUpload, samples }) {
  return (
    <div className="file-row">
      <div className="file-row-header">
        <span className="file-row-label">{label}</span>
        <span className="file-row-name">
          {file ? file.name : "No file selected"}
        </span>
      </div>
      <div className="file-row-actions">
        <label className="btn btn-outline">
          Choose file
          <input
            type="file"
            accept={accept}
            hidden
            onChange={(e) => onUpload(e.target.files[0])}
          />
        </label>
        {samples.map(({ name, path }) => (
          <button
            type="button"
            key={path}
            className="btn btn-ghost"
            onClick={async () => {
              try {
                onUpload(await fetchSample(path));
              } catch {
                // silently ignore — user will see "No file selected" remain
              }
            }}
          >
            {name}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [docFile, setDocFile] = useState(null);
  const [questionsFile, setQuestionsFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!docFile || !questionsFile) return;
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const form = new FormData();
      form.append("document_file", docFile);
      form.append("questions_file", questionsFile);

      const res = await fetch("/v1/qa", { method: "POST", body: form });
      const data = await res.json();

      if (!res.ok) {
        setError(data.detail ?? `Request failed (${res.status})`);
        return;
      }
      setResults(data.results);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const canSubmit = docFile && questionsFile && !loading;

  return (
    <div className="page">
      <header className="header">
        <h1>Document QA</h1>
        <p>
          Upload a document and a list of questions — answers grounded
          exclusively in the document.
        </p>
      </header>

      <main className="main">
        <form className="card upload-card" onSubmit={handleSubmit}>
          <FileRow
            label="Document"
            file={docFile}
            accept=".pdf,.json"
            onUpload={setDocFile}
            samples={[
              { name: "Sample PDF", path: "/samples/pdf_sample.pdf" },
              { name: "Sample JSON", path: "/samples/json_sample.json" },
            ]}
          />

          <div className="divider" />

          <FileRow
            label="Questions"
            file={questionsFile}
            accept=".json"
            onUpload={setQuestionsFile}
            samples={[
              { name: "PDF questions", path: "/samples/pdf_questions.json" },
              { name: "JSON questions", path: "/samples/json_questions.json" },
            ]}
          />

          <button
            className="btn btn-primary"
            type="submit"
            disabled={!canSubmit}
          >
            {loading ? (
              <>
                <span className="spinner" /> Analyzing…
              </>
            ) : (
              "Ask"
            )}
          </button>
        </form>

        {error && (
          <div className="card error-card" role="alert">
            <span>{error}</span>
            <button
              className="dismiss"
              onClick={() => setError(null)}
              aria-label="Dismiss"
            >
              ✕
            </button>
          </div>
        )}

        {results && (
          <section className="results">
            <h2 className="results-heading">
              {results.length} answer{results.length !== 1 ? "s" : ""}
            </h2>
            {results.map((r) => (
              <div key={r.id} className="card result-card">
                <p className="result-question">{r.question}</p>
                <p className="result-answer">{r.answer}</p>
                {r.sources.length > 0 && (
                  <div className="sources">
                    {r.sources.map((s, i) => (
                      <span key={i} className="source-chip">
                        {s.source}
                        {s.page != null && ` · p${s.page}`}
                        {s.json_path != null && ` · ${s.json_path}`}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </section>
        )}
      </main>
    </div>
  );
}
