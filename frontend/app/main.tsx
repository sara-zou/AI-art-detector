"use client";

import { useState, useRef } from "react";

export default function ArtifexDetector() {
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const analyse = async (file: File) => {
    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res  = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(`${data.label} — ${Math.round(data.confidence * 100)}% confidence`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div >
      <h1>AI Art Detector</h1>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={(e) => { if (e.target.files?.[0]) analyse(e.target.files[0]); }}
      />

      {loading && <p>Analysing…</p>}
      {result  && <p>Result: <strong>{result}</strong></p>}
      {error   && <p style={{ color: "red" }}>Error: {error}</p>}
    </div>
  );
}