import sys
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("🕵️ Person Detection using YOLOv8")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and convert image
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    # Detect using YOLO
    results = model.predict(img_np)[0]

    # Get detection results
    boxes = results.boxes
    class_names = model.names  # Dictionary: id → class name

    for box in boxes:
        cls_id = int(box.cls[0])
        label = class_names[cls_id]

        if label == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_np, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show result
    st.image(img_np, caption="Detected Person(s)", use_container_width=True)
