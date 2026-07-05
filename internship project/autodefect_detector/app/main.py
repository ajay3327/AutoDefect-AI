"""
Main Streamlit Application for AutoDefect AI.
"""

import os
import sys
import streamlit as st
import plotly.express as px

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.preprocessor import load_image_from_bytes, get_image_info, convert_color_space
from app.detector import load_model, load_settings, detect_and_parse
from app.annotator import annotate_image, draw_detection_summary
from app.estimator import calculate_damage_summary, get_cost_chart_data

@st.cache_resource
def cached_load_model(model_path: str):
    return load_model(model_path)

def cached_load_settings(config_path: str = None):
    return load_settings(config_path)

def setup_page():
    st.set_page_config(page_title="AutoDefect AI - Car Damage Detection", page_icon="🚗", layout="wide")

def main():
    setup_page()
    st.title("🚗 AutoDefect AI - Car Damage Detection")
    st.write("Intelligent Car Damage Detection powered by YOLOv8 Computer Vision")

    config_path = os.path.join(project_root, "config", "settings.json")
    # Copy the settings dict to safely modify it without mutating cached data
    settings = dict(cached_load_settings(config_path))

    # Sidebar settings
    st.sidebar.header("⚙️ Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, settings.get("model", {}).get("default_confidence", 0.25))

    st.sidebar.header("💰 Cost Parameters")
    cost_scratch = st.sidebar.number_input("Cost per Scratch (₹)", value=settings.get("cost_map", {}).get("Scratch", 1500))
    cost_dent = st.sidebar.number_input("Cost per Dent (₹)", value=settings.get("cost_map", {}).get("Dent", 3500))
    
    st.sidebar.header("🔑 Roboflow API Config")
    roboflow_cfg = settings.get("roboflow_api", {})
    api_key_input = st.sidebar.text_input(
        "Roboflow API Key", 
        value=roboflow_cfg.get("api_key", ""), 
        type="password",
        help="Leave empty to use local YOLO inference directly."
    )
    model_id_input = st.sidebar.text_input(
        "Model ID", 
        value=roboflow_cfg.get("model_id", ""),
        help="Roboflow model ID, e.g., project-name/version."
    )
    
    # Update settings with sidebar inputs
    if "roboflow_api" not in settings:
        settings["roboflow_api"] = {}
    settings["roboflow_api"]["api_key"] = api_key_input
    settings["roboflow_api"]["model_id"] = model_id_input

    user_params = {
        "conf_threshold": conf_threshold,
        "cost_map": {"Scratch": cost_scratch, "Dent": cost_dent}
    }

    # Load Model
    model_config = settings.get("model", {})
    model_path = model_config.get("default_model_path", "yolov8n.pt")
    custom_model_path = os.path.join(project_root, model_config.get("custom_model_path", ""))
    
    active_model_path = custom_model_path if os.path.exists(custom_model_path) else "yolov8n.pt"
    
    try:
        model = cached_load_model(active_model_path)
    except Exception as e:
        st.error(f"Failed to load the model: {str(e)}")
        st.stop()

    # Upload Section
    uploaded_file = st.file_uploader("Upload a car image for damage detection", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            file_bytes = uploaded_file.read()
            image_bgr = load_image_from_bytes(file_bytes)
            image_rgb = convert_color_space(image_bgr, "bgr2rgb")

            try:
                detections = detect_and_parse(
                    image=image_bgr,
                    model=model,
                    settings=settings,
                    conf_threshold=user_params["conf_threshold"],
                    use_model_names=True
                )
            except Exception as e:
                # If Roboflow cloud inference failed, fall back to the local YOLO model
                roboflow_cfg = settings.get("roboflow_api", {})
                if roboflow_cfg.get("api_key") and roboflow_cfg.get("api_key") != "API_KEY" and roboflow_cfg.get("model_id"):
                    st.warning(f"Roboflow API error: {str(e)}. Falling back to local YOLOv8 model.")
                    # Force fallback by passing settings with cleared API credentials
                    fallback_settings = settings.copy()
                    fallback_settings["roboflow_api"] = {"api_key": "", "model_id": ""}
                    try:
                        detections = detect_and_parse(
                            image=image_bgr,
                            model=model,
                            settings=fallback_settings,
                            conf_threshold=user_params["conf_threshold"],
                            use_model_names=True
                        )
                    except Exception as local_err:
                        st.error(f"Local Detection Error: {str(local_err)}")
                        detections = []
                else:
                    st.error(f"Detection Error: {str(e)}")
                    detections = []

        annotated_bgr = annotate_image(image_bgr, detections, settings.get("color_map", None))
        annotated_bgr = draw_detection_summary(annotated_bgr, detections)
        annotated_rgb = convert_color_space(annotated_bgr, "bgr2rgb")

        summary = calculate_damage_summary(
            detections,
            cost_map=user_params["cost_map"],
            severity_thresholds=settings.get("severity_thresholds", None)
        )

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Detections", summary["total_detections"])
        col2.metric("Scratches", summary["damage_counts"].get("Scratch", 0))
        col3.metric("Dents", summary["damage_counts"].get("Dent", 0))
        col4.metric("Repair Estimate", f"₹{summary['total_cost']:,.2f}")

        st.markdown("---")
        col_orig, col_anno = st.columns(2)
        with col_orig:
            st.image(image_rgb, caption="Original Image", use_container_width=True)
        with col_anno:
            st.image(annotated_rgb, caption="Detected Damages", use_container_width=True)

        if detections:
            chart_data = get_cost_chart_data(summary)
            if chart_data["class_names"]:
                col_bar, col_pie = st.columns(2)
                with col_bar:
                    fig_bar = px.bar(x=chart_data["class_names"], y=chart_data["costs"], labels={"x": "Type", "y": "Cost (₹)"}, title="Cost Breakdown")
                    st.plotly_chart(fig_bar, use_container_width=True)
                with col_pie:
                    fig_pie = px.pie(names=chart_data["class_names"], values=chart_data["counts"], title="Damage Distribution")
                    st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()
