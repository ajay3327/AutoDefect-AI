"""
Estimator Module for AutoDefect AI.

Calculates mock repair cost estimates and severity based on detections.
"""
from typing import List, Dict, Any

DEFAULT_COST_MAP = {"Scratch": 150, "Dent": 300}

def classify_severity(area_ratio: float, thresholds: Dict[str, float] = None) -> str:
    if thresholds is None:
        thresholds = {"minor_max_ratio": 0.02, "moderate_max_ratio": 0.05}

    if area_ratio < thresholds.get("minor_max_ratio", 0.02):
        return "Minor"
    elif area_ratio < thresholds.get("moderate_max_ratio", 0.05):
        return "Moderate"
    else:
        return "Severe"

def calculate_damage_summary(detections: List[Dict[str, Any]], cost_map: Dict[str, int] = None, severity_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    if cost_map is None:
        cost_map = DEFAULT_COST_MAP

    summary = {
        "total_detections": len(detections),
        "damage_counts": {},
        "total_cost": 0.0,
        "cost_by_class": {},
        "severity_counts": {"Minor": 0, "Moderate": 0, "Severe": 0},
        "details": []
    }

    for det in detections:
        cls = det.get("class_name", "Unknown")
        ratio = det.get("area_ratio", 0.0)
        
        severity = classify_severity(ratio, severity_thresholds)
        
        mult = {"Minor": 1.0, "Moderate": 1.5, "Severe": 2.5}.get(severity, 1.0)
        cost = round(cost_map.get(cls, 100) * mult, 2)

        summary["damage_counts"][cls] = summary["damage_counts"].get(cls, 0) + 1
        summary["cost_by_class"][cls] = summary["cost_by_class"].get(cls, 0.0) + cost
        summary["total_cost"] += cost
        summary["severity_counts"][severity] += 1
        
        summary["details"].append({
            "class_name": cls,
            "severity": severity,
            "confidence": det.get("confidence", 0.0),
            "cost": cost,
        })
        
    return summary

def get_cost_chart_data(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "class_names": list(summary["damage_counts"].keys()),
        "counts": list(summary["damage_counts"].values()),
        "costs": [summary["cost_by_class"].get(c, 0) for c in summary["damage_counts"].keys()]
    }
