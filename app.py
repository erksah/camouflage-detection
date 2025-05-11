import sys
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("🕵️🪖 Camouflage Person Detection using YOLOv8")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and convert image
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    # Detect using YOLO
    results = model.predict(img_np)[0]
    boxes = results.boxes
    class_names = model.names

    # Convert to PIL for drawing
    draw_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(draw_img)

    for box in boxes:
        cls_id = int(box.cls[0])
        label = class_names[cls_id]

        if label == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)
            draw.text((x1, y1 - 10), label, fill="green")

    # Show result with correct image display argument
    st.image(draw_img, caption="Detected Person(s)", use_column_width=True)
