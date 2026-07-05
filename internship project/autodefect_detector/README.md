# 🚗 AutoDefect AI — Intelligent Car Damage Detection Dashboard

AutoDefect AI is a modern, web-based computer vision application designed to automatically detect vehicle body defects (such as scratches and dents), classify the severity of the damage, and generate instant repair cost estimations. Powered by **YOLOv8**, **Streamlit**, and **OpenCV**, it supports both cloud-based **Roboflow Inference** and **Local PyTorch Inference** with auto-fallback capabilities.

---

## 🌟 Key Features

*   **Dual Inference Pipeline**:
    *   **Cloud Mode**: Connects to high-performance hosted models on Roboflow (using your API Key).
    *   **Local Fallback Mode**: Automatically falls back to a locally run YOLOv8 model (`models/damage_yolov8.pt`) if the cloud is offline or credentials are not supplied.
*   **Dynamic Severity Classification**: Analyzes the surface area ratio of defects to classify damage as **Minor (1.0x)**, **Moderate (1.5x)**, or **Severe (2.5x)**.
*   **Interactive Repair Estimator**: Live sidebar inputs to customize repair rates (e.g., base cost per scratch/dent) which dynamically updates the total repair estimate.
*   **Visual Analytics**: Integrates interactive **Plotly** charts showing damage category distributions and cost breakdowns.
*   **Flexible UI Configuration**: A polished Streamlit interface with sliding confidence thresholds, billing adjustments, and live API key management.

---

## 🛠️ Tech Stack

*   **Frontend & Dashboard**: [Streamlit](https://streamlit.io/)
*   **Computer Vision Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (PyTorch)
*   **Image Processing**: OpenCV, NumPy
*   **API Integrations**: Roboflow REST API
*   **Charts & Visualizations**: Plotly Express

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python 3.8+ installed. You can check your version by running:
```bash
python --version
```

### 2. Clone and Install
Clone this repository to your local machine and install the required dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/autodefect-detector.git
cd autodefect-detector
pip install -r requirements.txt
```

### 3. Model Weight Setup
*   **Local YOLO Nano Model**: The standard `yolov8n.pt` will automatically download on the first run.
*   **Custom Local Damage Model**: Place your fine-tuned `damage_yolov8.pt` weights in the `autodefect_detector/models/` directory to run high-accuracy local detections.

---

## 💻 Running the Application

Start the Streamlit dashboard using the runner script:
```bash
python run.py
```
*(Alternatively, you can run: `streamlit run app/main.py`)*

Open your browser and navigate to `http://localhost:8501`.

---

## ⚙️ Configuration & Credentials

You can run the app in two modes:

### A. Cloud Inference Mode (Roboflow)
1. Go to [Roboflow Universe](https://universe.roboflow.com/) and copy your **Private API Key** (starts with `uw...` or `07...`).
2. Paste it in the **Roboflow API Config** sidebar in the running app or save it directly in `config/settings.json`:
    ```json
    "roboflow_api": {
        "api_key": "YOUR_PRIVATE_API_KEY",
        "model_id": "car-damage-dn9sl/3"
    }
    ```

### B. Local Offline Mode
Simply leave the Roboflow API Key blank in the sidebar or set it to `""` in your `settings.json`. The app will automatically run inference locally on your CPU/GPU using your custom `damage_yolov8.pt` weights.

---

## 📁 Project Directory Structure

```text
autodefect_detector/
├── app/
│   ├── __init__.py
│   ├── annotator.py      # OpenCV image annotation & labels
│   ├── detector.py       # YOLOv8 local & Roboflow cloud API handlers
│   ├── estimator.py      # Severity calculation & cost estimation
│   ├── main.py           # Streamlit Dashboard UI and layout
│   └── preprocessor.py   # Image loading and formatting helpers
├── config/
│   └── settings.json     # Project settings, model paths, cost metrics
├── models/
│   ├── README.md
│   └── damage_yolov8.pt  # Local pre-trained damage model weights
├── tests/                # Automated PyTest suites
├── run.py                # Dashboard launcher script
└── requirements.txt      # Dependency list
```

---

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
