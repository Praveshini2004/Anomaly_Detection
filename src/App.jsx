
import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

function App() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [latestImages, setLatestImages] = useState({
    normal: [],
    warning: [],
    high: [],
  });

  const pollingRef = useRef(null);
  const BACKEND_URL = "http://localhost:5000";

  const fetchLatestImages = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/latest_images`);
      setLatestImages(response.data);
    } catch (error) {
      console.error("Error fetching latest images:", error);
    }
  };

  const startGeneration = async () => {
    setStatusMessage("Starting image generation...");
    setDownloadUrl("");
    setLatestImages({ normal: [], warning: [], high: [] });
    try {
      const response = await axios.post(`${BACKEND_URL}/start`);
      if (response.status === 200) {
        setIsGenerating(true);
        setStatusMessage("Image generation started. Images will update live.");
        fetchLatestImages();
        pollingRef.current = setInterval(fetchLatestImages, 10000);
      }
    } catch (error) {
      console.error("Error starting generation:", error);
      setStatusMessage("Failed to start image generation.");
    }
  };

  const stopGeneration = async () => {
    setStatusMessage("Stopping image generation...");
    try {
      const response = await axios.post(`${BACKEND_URL}/stop`);
      if (response.status === 200) {
        setIsGenerating(false);
        clearInterval(pollingRef.current);
        pollingRef.current = null;
        const downloadEndpoint = response.data.downloadUrl;
        setDownloadUrl(BACKEND_URL + downloadEndpoint);
        setStatusMessage("Generation stopped. Download available.");
        fetchLatestImages();
      }
    } catch (error) {
      console.error("Error stopping generation:", error);
      setStatusMessage("Failed to stop image generation.");
    }
  };

  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  const categoryColors = {
    normal: "#2563EB",
    warning: "#FBBF24",
    high: "#DC2626",
  };

  return (
    <div
      style={{
        fontFamily: "'Inter', sans-serif",
        maxWidth: 900,
        margin: "3rem auto",
        padding: "1.5rem",
        borderRadius: 12,
        boxShadow: "0 8px 20px rgba(0,0,0,0.1)",
        backgroundColor: "#fefefe",
      }}
      aria-live="polite"
      role="main"
    >
      <h1
        style={{ textAlign: "center", color: "#111827", marginBottom: "1.5rem" }}
      >
        CPU Usage Image Generator with Live Predictions
      </h1>

      <div style={{ display: "flex", justifyContent: "center", gap: 20, marginBottom: "1rem" }}>
        <button
          onClick={startGeneration}
          disabled={isGenerating}
          style={{
            backgroundColor: isGenerating ? "#9ca3af" : "#2563EB",
            color: "white",
            border: "none",
            padding: "0.75rem 2rem",
            fontSize: "1rem",
            borderRadius: "8px",
            cursor: isGenerating ? "not-allowed" : "pointer",
            transition: "background-color 0.3s ease",
          }}
          aria-disabled={isGenerating}
          aria-label="Start generating images"
        >
          Start
        </button>
        <button
          onClick={stopGeneration}
          disabled={!isGenerating}
          style={{
            backgroundColor: !isGenerating ? "#9ca3af" : "#DC2626",
            color: "white",
            border: "none",
            padding: "0.75rem 2rem",
            fontSize: "1rem",
            borderRadius: "8px",
            cursor: !isGenerating ? "not-allowed" : "pointer",
            transition: "background-color 0.3s ease",
          }}
          aria-disabled={!isGenerating}
          aria-label="Stop generating images"
        >
          Stop
        </button>
      </div>

      <div style={{ textAlign: "center", fontWeight: 600, marginBottom: "1rem", color: "#374151" }} aria-live="assertive">
        {statusMessage}
      </div>

      {isGenerating && (
        <section style={{ marginBottom: "2rem" }}>
          <h2 style={{ marginBottom: "1rem", textAlign: "center" }}>
            Live Image Preview & Predictions
          </h2>

          <div style={{ display: "flex", gap: 24, justifyContent: "center", flexWrap: "wrap" }}>
            {["normal", "warning", "high"].map((category) => (
              <div
                key={category}
                style={{
                  flex: "1 1 280px",
                  border: `2px solid ${categoryColors[category]}`,
                  borderRadius: 12,
                  padding: 12,
                  minHeight: 280,
                }}
              >
                <h3 style={{ textAlign: "center", marginBottom: 8, color: categoryColors[category], textTransform: "capitalize" }}>
                  {category} usage
                </h3>
                <div style={{ maxHeight: 230, overflowY: "auto" }}>
                  {latestImages[category] && latestImages[category].length > 0 ? (
                    latestImages[category].map(({ url, filename, prediction, confidence }) => (
                      <div key={filename} style={{ marginBottom: 12 }}>
                        <img
                          src={url}
                          alt={`CPU usage graph - ${category}`}
                          style={{
                            width: "100%",
                            borderRadius: 8,
                            boxShadow: "0 0 8px rgba(0,0,0,0.15)",
                          }}
                          loading="lazy"
                        />
                        {/* <div style={{ marginTop: 4, fontWeight: "600", textAlign: "center", color: categoryColors[category] }}>
                          Prediction: {prediction} ({(confidence * 100).toFixed(1)}%)
                        </div> */}
                      </div>
                    ))
                  ) : (
                    <p style={{ textAlign: "center", color: "#6B7280" }}>
                      No images yet
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {downloadUrl && (
        <div style={{ textAlign: "center" }}>
          <a
            href={downloadUrl}
            download
            style={{
              display: "inline-block",
              padding: "0.75rem 2rem",
              backgroundColor: "#10B981",
              color: "white",
              borderRadius: "8px",
              fontWeight: "bold",
              textDecoration: "none",
              transition: "background-color 0.3s ease",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#059669")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#10B981")}
            aria-label="Download zip file of CPU usage images"
          >
            Download Images
          </a>
        </div>
      )}

      <footer style={{ marginTop: "3rem", fontSize: "0.875rem", color: "#6B7280", textAlign: "center", userSelect: "none" }}>
        Powered by React, Vite.js & Flask
      </footer>
    </div>
  );
}

export default App;

