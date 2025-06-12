// src/ImageGenerator.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ImageGenerator = () => {
    const [image, setImage] = useState(null);
    const [isGenerating, setIsGenerating] = useState(false);
    const [prediction, setPrediction] = useState('');

    const generateRandomImage = () => {
        // Generate a random image (you can customize this logic)
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 128;
        canvas.height = 128;

        // Fill the canvas with random colors
        for (let i = 0; i < 128; i++) {
            for (let j = 0; j < 128; j++) {
                ctx.fillStyle = `rgb(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255})`;
                ctx.fillRect(i, j, 1, 1);
            }
        }

        return canvas.toDataURL('image/png');
    };

    const predictImage = async (imageData) => {
        try {
            const response = await axios.post('http://localhost:5000/predict', { image: imageData });
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error('Error predicting image:', error);
        }
    };

    const startGenerating = () => {
        setIsGenerating(true);
        const interval = setInterval(() => {
            const randomImage = generateRandomImage();
            setImage(randomImage);
            predictImage(randomImage);
        }, 1000);

        // Store the interval ID to clear it later
        return interval;
    };

    const stopGenerating = (intervalId) => {
        clearInterval(intervalId);
        setIsGenerating(false);
    };

    useEffect(() => {
        let intervalId;
        if (isGenerating) {
            intervalId = startGenerating();
        }
        return () => {
            if (intervalId) {
                stopGenerating(intervalId);
            }
        };
    }, [isGenerating]);

    return (
        <div>
            <h1>Random Image Generator</h1>
            {image && <img src={image} alt="Random" />}
            <div>
                <button onClick={() => setIsGenerating(true)}>Start Generating</button>
                <button onClick={() => stopGenerating()}>Stop</button>
            </div>
            {prediction && <h2>Prediction: {prediction}</h2>}
        </div>
    );
};

export default ImageGenerator;
