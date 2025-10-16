import streamlit as st
from src.yolo_inference import load_model, run_inference
from src.dedupe import dedupe_per_room
from src.report import build_report
from src.utils import to_numpy, draw_boxes_pil, pil_bytes_from_numpy
from PIL import Image
import io
import os
import pandas as pd

# class names from your yaml (index -> name)
CLASS_NAMES = {
    0: "bed",
    1: "sofa",
    2: "chair",
    3: "table",
    4: "lamp",
    5: "tv",
    6: "laptop",
    7: "wardrobe",
    8: "window",
    9: "door",
    10: "potted plant",
    11: "photo frame"
}
CLASS_ORDER = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]

st.set_page_config(page_title="Room Object Detector", layout="wide")

st.title("Room Object Detection — Streamlit")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value="models/best.pt")
    device = st.selectbox("Device", options=["cpu", "cuda"], index=0 if not st.session_state.get("cuda_available", False) else 1)
    conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    imgsz = st.slider("Image size for inference", 320, 1280, 640, step=32)
    st.markdown("Note: annotated images are displayed but not saved to disk.")
    st.write("") 

uploaded = st.file_uploader("Upload room images (each image = one room)", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded:
    # preview and allow edit of room names
    st.subheader("Uploaded Files")
    room_names = []
    cols = st.columns(4)
    for i, file in enumerate(uploaded):
        default_name = os.path.splitext(file.name)[0]
        col = cols[i % 4]
        with col:
            st.image(file, use_column_width=True)
            rn = st.text_input(f"Room name #{i+1}", value=default_name, key=f"roomname_{i}")
            room_names.append(rn)
    run = st.button("Run Detection")
    if run:
        # load model
        try:
            model = load_model(model_path, device=device)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

        room2classes = {}
        annotated_images = {}
        no_detections = []
        progress = st.progress(0)
        total = len(uploaded)
        for i, file in enumerate(uploaded):
            try:
                img = Image.open(io.BytesIO(file.read())).convert("RGB")
            except Exception:
                st.warning(f"Could not open {file.name}")
                continue
            np_img = to_numpy(img)
            detections = run_inference(model, np_img, conf_thres=conf_thresh, imgsz=imgsz)
            dedup = dedupe_per_room(detections, CLASS_NAMES, conf_thresh=conf_thresh)
            room_id = room_names[i] if i < len(room_names) else os.path.splitext(file.name)[0]
            room2classes[room_id] = dedup
            if len(detections) == 0 or len(dedup) == 0:
                no_detections.append(room_id)
            # draw boxes for display
            annotated_np = draw_boxes_pil(np_img, detections, CLASS_NAMES)
            annotated_images[room_id] = pil_bytes_from_numpy(annotated_np)
            progress.progress((i+1)/total)
        progress.empty()

        # build DataFrame
        df = build_report(room2classes, CLASS_ORDER)
        st.subheader("Per-room & Total Summary")
        if df is None or df.empty:
            st.info("No detections found in any uploaded images.")
        else:
            st.dataframe(df.astype(int))

            # CSV download
            csv_bytes = df.astype(int).to_csv().encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="room_report.csv", mime="text/csv")

        # show messages for rooms with no detections
        if no_detections:
            st.warning("No objects detected in: " + ", ".join(no_detections))

        st.subheader("Annotated Images")
        for room_id, img_bytes in annotated_images.items():
            st.image(img_bytes, caption=room_id, use_column_width=True)
