'use client'
import React, { useState } from "react";

export function FileUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [segmentedImage, setSegmentedImage] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("http://localhost:5000/upload", {
      method: "POST",
      body: formData,
    });

    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);
    setSegmentedImage(imageUrl);
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h2>Upload MRI Scan (PNG)</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/png" onChange={handleFileChange} />
        <button type="submit">Submit</button>
      </form>

      {segmentedImage && (
        <div>
          <h3>Segmented Result</h3>
          <img src={segmentedImage} alt="Segmented MRI" style={{ maxWidth: "100%" }} />
        </div>
      )}
    </div>
  );
}
