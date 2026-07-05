"""
Detector Module for AutoDefect AI.

Wraps the YOLOv8 model for object detection:
  - Loading YOLO weights.
  - Running inference on images.
  - Parsing detection results into dictionaries.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

def load_model(model_path: str, device: str = "auto") -> YOLO:
    standard_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    model_filename = os.path.basename(model_path)
    
    if model_filename not in standard_models and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: '{model_path}'.")

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{model_path}': {str(e)}")

def run_inference(model: YOLO, image: np.ndarray, conf_threshold: float = 0.25, device: str = "auto") -> Any:
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = model.predict(source=image, conf=conf_threshold, device=device, verbose=False)
    return results

def load_settings(config_path: str = None) -> Dict:
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "settings.json")
    
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, "r") as f:
        return json.load(f)

def get_class_name(cls_id: int, settings: Dict) -> str:
    class_names = settings.get("class_names", {})
    return class_names.get(str(cls_id), "Unknown")

def parse_detections(results: Any, settings: Dict, image_shape: Tuple[int, int] = None, use_model_names: bool = True) -> List[Dict[str, Any]]:
    detections = []
    result = results[0] if isinstance(results, list) else results
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return detections

    total_image_area = image_shape[0] * image_shape[1] if image_shape else 1

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy.tolist()
        confidence = round(box.conf[0].cpu().item(), 2)
        cls_id = int(box.cls[0].cpu().item())

        if use_model_names and hasattr(result, 'names') and result.names:
            class_name = result.names.get(cls_id, "Unknown")
        else:
            class_name = get_class_name(cls_id, settings)

        if class_name:
            class_name = class_name.replace("_", " ").title().replace(" ", "_")

        box_area = abs(x2 - x1) * abs(y2 - y1)
        area_ratio = round(box_area / total_image_area, 4) if total_image_area > 0 else 0

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
            "class_id": cls_id,
            "class_name": class_name,
            "box_area": box_area,
            "area_ratio": area_ratio
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections

def run_roboflow_inference(image: np.ndarray, api_key: str, model_id: str, conf_threshold: float, settings: Dict) -> List[Dict[str, Any]]:
    import cv2
    import base64
    import requests
    
    # Convert the OpenCV image directly to a base64 string
    _, buffer = cv2.imencode('.jpg', image)
    img_b64 = base64.b64encode(buffer).decode("utf-8")
    
    # Make a direct REST API call to Roboflow (Bypasses the inference-sdk version block!)
    api_url = f"https://detect.roboflow.com/{model_id}?api_key={api_key}&confidence={conf_threshold}"
    
    response = requests.post(
        api_url, 
        data=img_b64, 
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"Roboflow API Error: {response.text}")
        
    result = response.json()
            
    detections = []
    total_image_area = image.shape[0] * image.shape[1]
    
    predictions = result.get("predictions", [])
    for pred in predictions:
        conf = pred.get("confidence", 0)
        if conf < conf_threshold:
            continue
            
        x_center = pred["x"]
        y_center = pred["y"]
        w = pred["width"]
        h = pred["height"]
        
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
        
        class_name = pred.get("class", "Unknown")
        if class_name:
            class_name = class_name.replace("_", " ").title().replace(" ", "_")
        cls_id = pred.get("class_id", 0)
        
        box_area = w * h
        area_ratio = round(box_area / total_image_area, 4) if total_image_area > 0 else 0
        
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": round(conf, 2),
            "class_id": cls_id,
            "class_name": class_name,
            "box_area": box_area,
            "area_ratio": area_ratio
        })
        
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections

def detect_and_parse(image: np.ndarray, model: YOLO, settings: Dict, conf_threshold: float = 0.25, use_model_names: bool = True) -> List[Dict[str, Any]]:
    roboflow_cfg = settings.get("roboflow_api", {})
    api_key = os.environ.get("ROBOFLOW_API_KEY") or roboflow_cfg.get("api_key", "")
    model_id = os.environ.get("ROBOFLOW_MODEL_ID") or roboflow_cfg.get("model_id", "")
    
    # If the user has added a valid API key, use Roboflow Cloud
    if api_key and api_key != "API_KEY" and model_id:
        return run_roboflow_inference(image, api_key, model_id, conf_threshold, settings)
        
    # Otherwise, fall back to local YOLO model
    results = run_inference(model, image, conf_threshold=conf_threshold)
    return parse_detections(results, settings, image_shape=image.shape[:2], use_model_names=use_model_names)
