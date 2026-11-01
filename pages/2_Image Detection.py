import os
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO
from PIL import Image, ImageOps
from io import BytesIO

from sample_utils.download import download_file

st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Session-specific caching
# Load the model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

st.title("Road Damage Detection - Image")
st.write("Detect the road damage in using an Image input. Upload the image and start detecting. This section can be useful for examining baseline data.")

image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg', 'webp', 'bmp'])

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
st.write("Lower the threshold if there is no damage detected, and increase the threshold if there is false prediction.")

if image_file is not None:

    # Load the image
    # Normalise to a 3-channel RGB image (handles grayscale, RGBA/PNG
    # transparency, palette images and phone-camera EXIF rotation)
    image = Image.open(image_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    
    col1, col2 = st.columns(2)

    # Perform inference
    _image = np.array(image)
    h_ori = _image.shape[0]
    w_ori = _image.shape[1]

    image_resized = cv2.resize(_image, (640, 640), interpolation = cv2.INTER_AREA)
    # Ultralytics expects BGR numpy arrays
    results = net.predict(cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR), conf=score_threshold)
    
    # Save the results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
           Detection(
               class_id=int(_box.cls),
               label=CLASSES[int(_box.cls)],
               score=float(_box.conf),
               box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]

    annotated_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation = cv2.INTER_AREA)

    # Summary of what the model found
    all_detections = [d for r in results for d in
                      (Detection(int(b.cls), CLASSES[int(b.cls)], float(b.conf), b.xyxy[0].astype(int))
                       for b in r.boxes.cpu().numpy())]
    if all_detections:
        st.success(f"✅ {len(all_detections)} damage region(s) detected at confidence ≥ {score_threshold:.2f}")
        st.table([
            {"#": i + 1, "Damage Type": d.label, "Confidence": f"{d.score:.0%}"}
            for i, d in enumerate(sorted(all_detections, key=lambda d: -d.score))
        ])
    else:
        st.warning(
            f"⚠️ No damage detected at confidence ≥ {score_threshold:.2f}. "
            "The prediction image is identical to the original. Lower the threshold slider to look for weaker detections."
        )

    # Original Image
    with col1:
        st.markdown("#### 🖼️ Original Image")
        st.caption("Uploaded photo, no processing")
        st.image(_image)
    
    # Predicted Image
    with col2:
        st.markdown("#### 🎯 Model Prediction")
        st.caption("Detected damage drawn as coloured boxes with confidence")
        st.image(_image_pred)

        # Download predicted image
        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save(buffer, format="PNG")
        _downloadImagesByte = buffer.getvalue()

        downloadButton = st.download_button(
            label="Download Prediction Image",
            data=_downloadImagesByte,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )