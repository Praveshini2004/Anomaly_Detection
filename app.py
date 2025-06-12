


import os
import time
import threading
import zipfile
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# === Flask app setup ===
app = Flask(__name__)
CORS(app)

# === Configuration ===
BASE_DIR = "cpu_usage_images_generated"
NORMAL_DIR = os.path.join(BASE_DIR, "normal")
WARNING_DIR = os.path.join(BASE_DIR, "warning")
HIGH_DIR = os.path.join(BASE_DIR, "high")
ZIP_FILENAME = "cpu_usage_multi_category_graphs.zip"

DURATION = 30
ANOMALY_THRESHOLD = 70

# Create dirs
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(WARNING_DIR, exist_ok=True)
os.makedirs(HIGH_DIR, exist_ok=True)

# === Thread control ===
generation_thread = None
stop_event = threading.Event()
lock = threading.Lock()

# === Model definition & loading ===
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.fc1 = torch.nn.Linear(128 * 16 * 16, 128)
        self.fc2 = torch.nn.Linear(128, 3)  # classes: normal, warning, high

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cpu")
model = SimpleCNN()
model.load_state_dict(torch.load("server_detection_model.pth", map_location=device))
model.eval()

inference_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

category_names = ["normal", "warning", "high"]

def predict_image_category(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        input_tensor = inference_transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            return {
                "category": category_names[pred_idx.item()],
                "confidence": round(conf.item(), 3)
            }
    except Exception as e:
        print(f"Prediction error on {image_path}: {e}")
        return {"category": "unknown", "confidence": 0.0}

# === CPU utilization data generation ===
def generate_utilization(category: str) -> np.ndarray:
    time_arr = np.arange(DURATION)
    ideal_min, ideal_max = 30, 70

    baseline = ideal_min + (ideal_max - ideal_min) * (0.5 + 0.4 * np.sin(2 * np.pi * time_arr / DURATION * 3))
    noise = np.random.normal(0, 3, size=DURATION)
    utilization = baseline + noise
    utilization = np.clip(utilization, 0, 100)

    if category == "normal":
        n_spikes = np.random.randint(0, 3)
        for _ in range(n_spikes):
            spike_len = np.random.randint(1, 4)
            spike_start = np.random.randint(0, DURATION - spike_len)
            spike_height = np.random.uniform(ANOMALY_THRESHOLD - 5, ANOMALY_THRESHOLD + 5)
            spike_center = spike_start + spike_len // 2
            spike_range = np.arange(spike_start, spike_start + spike_len)
            gaussian_spike = spike_height * np.exp(-0.5 * ((spike_range - spike_center) / (spike_len / 3))**2)
            utilization[spike_range] = np.maximum(utilization[spike_range], gaussian_spike)
    elif category == "warning":
        n_spikes = np.random.randint(1, 4)
        for _ in range(n_spikes):
            spike_len = np.random.randint(3, 8)
            spike_start = np.random.randint(0, DURATION - spike_len)
            spike_height = np.random.uniform(ANOMALY_THRESHOLD + 1, ANOMALY_THRESHOLD + 15)
            spike_center = spike_start + spike_len // 2
            spike_range = np.arange(spike_start, spike_start + spike_len)
            gaussian_spike = spike_height * np.exp(-0.5 * ((spike_range - spike_center) / (spike_len / 3))**2)
            utilization[spike_range] = np.maximum(utilization[spike_range], gaussian_spike)
    else:  # high usage
        n_spikes = np.random.randint(4, 11)
        for _ in range(n_spikes):
            spike_len = np.random.randint(3, 8)
            spike_start = np.random.randint(0, DURATION - spike_len)
            spike_height = np.random.uniform(ANOMALY_THRESHOLD + 10, 100)
            spike_center = spike_start + spike_len // 2
            spike_range = np.arange(spike_start, spike_start + spike_len)
            gaussian_spike = spike_height * np.exp(-0.5 * ((spike_range - spike_center) / (spike_len / 3))**2)
            utilization[spike_range] = np.maximum(utilization[spike_range], gaussian_spike)

    utilization = np.clip(utilization, 0, 100)
    return utilization

def plot_and_save(index: int, utilization: np.ndarray, label: str) -> str:
    time_arr = np.arange(1, DURATION + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(time_arr, utilization, label='CPU Utilization', color='#2563EB', marker='o', markersize=5, linewidth=2)
    plt.axhline(y=ANOMALY_THRESHOLD, color='#DC2626', linestyle='--', linewidth=2, label='Anomaly Threshold')
    plt.fill_between(time_arr, ANOMALY_THRESHOLD, utilization, where=(utilization > ANOMALY_THRESHOLD),
                     color='#FCA5A5', alpha=0.4, label='Anomalies')

    plt.xlabel('Time (seconds)', fontsize=14, color='#374151')
    plt.ylabel('CPU Utilization (%)', fontsize=14, color='#374151')
    plt.title('CPU Server Utilization Over One Minute', fontsize=18, weight='bold', color='#111827')
    plt.legend(fontsize=12)
    plt.grid(True, color='#D1D5DB')
    plt.ylim(0, 110)
    plt.xlim(1, DURATION)
    plt.tight_layout()
    plt.gca().set_facecolor('#FFFFFF')

    folder_map = {
        "normal": NORMAL_DIR,
        "warning": WARNING_DIR,
        "high": HIGH_DIR,
    }
    folder = folder_map[label]
    filename = f"{label}_{index:04d}.png"
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath

def clear_all_images():
    for folder in [NORMAL_DIR, WARNING_DIR, HIGH_DIR]:
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            if os.path.isfile(path):
                os.remove(path)

def generate_images_continuously():
    index_counters = {"normal": 0, "warning": 0, "high": 0}
    categories = ["normal", "warning", "high"]

    while not stop_event.is_set():
        with lock:
            for label in categories:
                util = generate_utilization(label)
                plot_and_save(index_counters[label], util, label)
                index_counters[label] += 1
        for _ in range(6):  # 5 seconds / 1 second = 5 iterations
            if stop_event.is_set():
                break
            time.sleep(5)  # Change sleep time to 5 seconds


def create_zip_file():
    zip_path = os.path.join(BASE_DIR, ZIP_FILENAME)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for category, folder in {
            "normal": NORMAL_DIR,
            "warning": WARNING_DIR,
            "high": HIGH_DIR,
        }.items():
            for file_name in os.listdir(folder):
                file_path = os.path.join(folder, file_name)
                arcname = os.path.join(category, file_name)
                zipf.write(file_path, arcname)
    return zip_path

@app.route("/start", methods=["POST"])
def start_generation():
    global generation_thread, stop_event
    with lock:
        if generation_thread and generation_thread.is_alive():
            return jsonify({"message": "Generation already running"}), 400
        clear_all_images()
        stop_event.clear()
        generation_thread = threading.Thread(target=generate_images_continuously)
        generation_thread.start()
    return jsonify({"message": "Image generation started"}), 200

@app.route("/stop", methods=["POST"])
def stop_generation():
    global generation_thread, stop_event
    with lock:
        if not generation_thread or not generation_thread.is_alive():
            return jsonify({"message": "No generation running"}), 400
        stop_event.set()
        generation_thread.join()
        generation_thread = None
        create_zip_file()
    return jsonify({"downloadUrl": "/download_zip"}), 200

@app.route("/images/<category>/<filename>")
def serve_image(category, filename):
    folder_map = {
        "normal": NORMAL_DIR,
        "warning": WARNING_DIR,
        "high": HIGH_DIR,
    }
    folder = folder_map.get(category)
    if folder:
        return send_from_directory(folder, filename)
    else:
        return "Invalid category", 404

@app.route("/latest_images")
def latest_images():
    num_latest = 5
    response = {}
    for category, folder in {
        "normal": NORMAL_DIR,
        "warning": WARNING_DIR,
        "high": HIGH_DIR,
    }.items():
        try:
            files = [f for f in os.listdir(folder) if f.endswith(".png")]
            files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
            latest_files = files[:num_latest]
            base_url = f"http://localhost:5000/images/{category}/"
            images_info = []
            for fname in latest_files:
                full_path = os.path.join(folder, fname)
                mtime = int(os.path.getmtime(full_path))
                url = f"{base_url}{fname}?v={mtime}"
                pred = predict_image_category(full_path)
                images_info.append({
                    "url": url,
                    "filename": fname,
                    "prediction": pred["category"],
                    "confidence": pred["confidence"],
                })
            response[category] = images_info
        except Exception as e:
            print(f"Error fetching images for {category}: {e}")
            response[category] = []
    return jsonify(response)

@app.route("/download_zip")
def download_zip():
    return send_from_directory(BASE_DIR, ZIP_FILENAME, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
