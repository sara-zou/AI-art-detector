"use client";

import { useState, useRef, DragEvent } from "react";

export default function ArtifexDetector() {
  const [result, setResult]   = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    const url = URL.createObjectURL(file);
    setPreview(url);
    analyse(file);
  };

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
      setResult(`${data.label}, ${(data.confidence * 100).toFixed(2)}% confidence`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const onDragOver  = (e: DragEvent) => { e.preventDefault(); setIsDragOver(true); };
  const onDragLeave = () => setIsDragOver(false);
  const onDrop      = (e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div>
      <h1>AI Art Detector</h1>

      <div
        onClick={() => fileInputRef.current?.click()}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        style={{
          border: `2px dashed ${isDragOver ? "blue" : "gray"}`,
          borderRadius: "8px",
          cursor: "pointer",
          background: isDragOver ? "#f0f4ff" : "transparent",
          transition: "all 0.2s",
          position: "relative",
          width: "400px",
          height: "300px",
          overflow: "hidden",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {preview ? (
          <img
            src={preview}
            alt="Uploaded preview"
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <p style={{ color: "gray" }}>Drop an image here or click to browse</p>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }}
      />

      {loading && <p>Analysing…</p>}
      {result  && <p>Result: <strong>{result}</strong></p>}
      {error   && <p style={{ color: "red" }}>Error: {error}</p>}
    </div>
  );
}