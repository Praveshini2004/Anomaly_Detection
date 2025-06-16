
import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import './App.css'; 

function ImageGenerator() {
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
    <div className="container" aria-live="polite" role="main">
      <h1 className="header">CPU Usage Image Generator with Live Predictions</h1>

      <div className="button-container">
        <button
          onClick={startGeneration}
          disabled={isGenerating}
          className={`button start`}
          aria-disabled={isGenerating}
          aria-label="Start generating images"
        >
          Start
        </button>
        <button
          onClick={stopGeneration}
          disabled={!isGenerating}
          className={`button stop`}
          aria-disabled={!isGenerating}
          aria-label="Stop generating images"
        >
          Stop
        </button>
      </div>

      <div className="status-message" aria-live="assertive">
        {statusMessage}
      </div>

      {isGenerating && (
        <section className="image-preview-section">
          <h2 className="image-preview-title">Live Image Preview & Predictions</h2>

          <div style={{ display: "flex", gap: 24, justifyContent: "center", flexWrap: "wrap" }}>
            {["normal", "warning", "high"].map((category) => (
              <div
                key={category}
                className="image-category"
                style={{ border: `2px solid ${categoryColors[category]}` }}
              >
                <h3 style={{ color: categoryColors[category] }}>{category} usage</h3>
                <div className="image-list">
                  {latestImages[category] && latestImages[category].length > 0 ? (
                    latestImages[category].map(({ url, filename }) => (
                      <div key={filename} className="image-item">
                        <img
                          src={url}
                          alt={`CPU usage graph - ${category}`}
                          loading="lazy"
                        />
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
        <div className="download-link">
          <a
            href={downloadUrl}
            download
            className="download-button"
            aria-label="Download zip file of CPU usage images"
          >
            Download Images
          </a>
        </div>
      )}
    </div>
  );
}

export default ImageGenerator;


