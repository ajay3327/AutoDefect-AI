"""
Annotator Module for AutoDefect AI.

Draws bounding boxes and labels on images using OpenCV.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

DEFAULT_COLORS = {
    "Scratch": (0, 255, 255),
    "Dent": (0, 0, 255),
    "Unknown": (0, 255, 0),
}

def get_color_for_class(class_name: str, color_map: Dict[str, List[int]] = None) -> Tuple[int, int, int]:
    if color_map and class_name in color_map:
        return tuple(color_map[class_name])
    return DEFAULT_COLORS.get(class_name, DEFAULT_COLORS["Unknown"])

def draw_single_box(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, color: Tuple[int, int, int], confidence: float) -> np.ndarray:
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    
    label_text = f"{label} {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    (text_w, text_h), _ = cv2.getTextSize(label_text, font, 0.7, 2)
    cv2.rectangle(image, (x1, max(0, y1 - text_h - 10)), (x1 + text_w + 10, y1), color, -1)
    
    cv2.putText(image, label_text, (x1 + 5, y1 - 5), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def annotate_image(image: np.ndarray, detections: List[Dict[str, Any]], color_map: Dict[str, List[int]] = None) -> np.ndarray:
    annotated = image.copy()
    if not detections:
        return annotated

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = get_color_for_class(det["class_name"], color_map)
        annotated = draw_single_box(annotated, x1, y1, x2, y2, det["class_name"], color, det["confidence"])
        
    return annotated

def draw_detection_summary(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    annotated = image.copy()
    if not detections:
        return annotated

    counts = {}
    for det in detections:
        counts[det["class_name"]] = counts.get(det["class_name"], 0) + 1

    y_offset = 20
    for name, count in counts.items():
        text = f"{name}: {count}"
        cv2.putText(annotated, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 25
        
    return annotated
