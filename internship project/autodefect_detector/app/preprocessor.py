"""
Preprocessor Module for AutoDefect AI.

Handles image ingestion and preprocessing tasks:
  - Loading images from bytes.
  - Extracting image metadata.
  - Resizing images to target sizes.
  - Color space conversions.
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, Dict

def load_image_from_bytes(byte_data: bytes) -> np.ndarray:
    try:
        pil_image = Image.open(BytesIO(byte_data))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        img_rgb = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        raise ValueError(f"Failed to decode image from byte stream: {str(e)}")

def get_image_info(image: np.ndarray) -> Dict[str, int]:
    if len(image.shape) == 3:
        height, width, channels = image.shape
    elif len(image.shape) == 2:
        height, width = image.shape
        channels = 1
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    return {
        "height": height,
        "width": width,
        "channels": channels,
        "total_pixels": height * width
    }

def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def convert_color_space(image: np.ndarray, conversion: str = "bgr2rgb") -> np.ndarray:
    conversion_map = {
        "bgr2rgb": cv2.COLOR_BGR2RGB,
        "rgb2bgr": cv2.COLOR_RGB2BGR,
    }
    if conversion.lower() not in conversion_map:
        raise ValueError(f"Unsupported conversion: '{conversion}'")
    return cv2.cvtColor(image, conversion_map[conversion.lower()])

def preprocess_for_model(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    resized = resize_image(image, target_size=target_size)
    rgb_image = convert_color_space(resized, "bgr2rgb")
    return rgb_image
