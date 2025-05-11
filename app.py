import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# Load YOLOv8 model (replace with "camouflage.pt" if you have a custom one)
model = YOLO("yolov8n.pt")  # or "camouflage.pt" if trained

st.title("🪖 Camouflage Person Detection using YOLOv8")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and convert image
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    # Run detection
    results = model.predict(img_np, verbose=False)[0]
    boxes = results.boxes
    class_names = model.names

    # Draw on image using PIL
    draw_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(draw_img)

    for box in boxes:
        cls_id = int(box.cls[0])
        label = class_names[cls_id]

        if label == "person":  # Only show persons
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)
            draw.text((x1, y1 - 10), label, fill="green")

    # Show result
    st.image(draw_img, caption="Detected Person(s)", use_column_width=True)
