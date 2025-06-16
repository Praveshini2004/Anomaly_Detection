import matplotlib
matplotlib.use('Agg')  # Non-GUI backend

import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile

# === Configuration ===
base_dir = "cpu_usage_images"
normal_dir = os.path.join(base_dir, "normal")
warning_dir = os.path.join(base_dir, "warning")
high_dir = os.path.join(base_dir, "high")
zip_filename = "cpu_usage_multi_category_graphs.zip"
num_images = 3000  # total images
duration = 30  # seconds per plot
ideal_min = 30
ideal_max = 70
anomaly_threshold = 70
sustained_high_duration = 5  # minimum seconds for sustained high CPU usage

# Create directories
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(warning_dir, exist_ok=True)
os.makedirs(high_dir, exist_ok=True)

def generate_utilization(category: str) -> np.ndarray:
    time = np.arange(duration)
    baseline = ideal_min + (ideal_max - ideal_min) * (0.5 + 0.4 * np.sin(2 * np.pi * time / duration * 3))
    noise = np.random.normal(0, 3, size=duration)
    utilization = baseline + noise
    utilization = np.clip(utilization, 0, 100)

    if category == "normal":
        # Few mild spikes just below or around threshold
        n_spikes = np.random.randint(0, 3)
        for _ in range(n_spikes):
            spike_len = np.random.randint(1, 3)
            spike_start = np.random.randint(0, duration - spike_len)
            spike_height = np.random.uniform(anomaly_threshold - 5, anomaly_threshold + 2)
            spike_center = spike_start + spike_len // 2
            spike_range = np.arange(spike_start, spike_start + spike_len)
            gaussian_spike = spike_height * np.exp(-0.5 * ((spike_range - spike_center) / (spike_len / 3)) ** 2)
            utilization[spike_range] = np.maximum(utilization[spike_range], gaussian_spike)

    elif category == "warning":
        # Short spikes over threshold
        n_spikes = np.random.randint(1, 4)
        for _ in range(n_spikes):
            spike_len = np.random.randint(2, 5)  # less than 5 sec
            spike_start = np.random.randint(0, duration - spike_len)
            spike_height = np.random.uniform(anomaly_threshold + 1, anomaly_threshold + 15)
            spike_center = spike_start + spike_len // 2
            spike_range = np.arange(spike_start, spike_start + spike_len)
            gaussian_spike = spike_height * np.exp(-0.5 * ((spike_range - spike_center) / (spike_len / 3)) ** 2)
            utilization[spike_range] = np.maximum(utilization[spike_range], gaussian_spike)

    elif category == "high":
        # At least one spike > threshold lasting >= 5 seconds
        n_spikes = np.random.randint(2, 5)
        for _ in range(n_spikes):
            spike_len = np.random.randint(sustained_high_duration, 8)  # sustained spike
            spike_start = np.random.randint(0, duration - spike_len)
            spike_height = np.random.uniform(anomaly_threshold + 10, 100)
            spike_center = spike_start + spike_len // 2
            spike_range = np.arange(spike_start, spike_start + spike_len)
            gaussian_spike = spike_height * np.exp(-0.5 * ((spike_range - spike_center) / (spike_len / 3)) ** 2)
            utilization[spike_range] = np.maximum(utilization[spike_range], gaussian_spike)

    utilization = np.clip(utilization, 0, 100)
    return utilization

def plot_and_save(index: int, utilization: np.ndarray, label: str) -> None:
    time = np.arange(1, duration + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(time, utilization, label='CPU Utilization', color='#2563EB', marker='o', markersize=4, linewidth=2)
    plt.axhline(y=anomaly_threshold, color='#DC2626', linestyle='--', linewidth=2, label='Anomaly Threshold')
    plt.fill_between(time, anomaly_threshold, utilization, where=(utilization > anomaly_threshold),
                     color='#FCA5A5', alpha=0.4, label='Anomalies')

    plt.xlabel('Time (seconds)', fontsize=14, color='#374151')
    plt.ylabel('CPU Utilization (%)', fontsize=14, color='#374151')
    plt.title('CPU Server Utilization Over One Minute', fontsize=18, weight='bold', color='#111827')
    plt.legend(fontsize=12)
    plt.grid(True, color='#D1D5DB')
    plt.ylim(0, 110)
    plt.xlim(1, duration)
    plt.tight_layout()
    plt.gca().set_facecolor('#FFFFFF')

    folder_map = {
        "normal": normal_dir,
        "warning": warning_dir,
        "high": high_dir
    }
    folder = folder_map[label]
    filename = f"{label}_{index:04d}.png"
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath)
    plt.close()

def main():
    np.random.seed(42)
    categories = ["normal", "warning", "high"]
    images_per_category = num_images // len(categories)
    index = 0

    for label in categories:
        for _ in range(images_per_category):
            utilization = generate_utilization(label)
            plot_and_save(index, utilization, label)
            index += 1

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for label in categories:
            folder_path = os.path.join(base_dir, label)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                arcname = os.path.join(label, file_name)
                zipf.write(file_path, arcname)

    print(f"✅ ZIP file created: {os.path.abspath(zip_filename)}")

if __name__ == "__main__":
    main()