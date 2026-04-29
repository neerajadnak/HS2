import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# Page config
st.set_page_config(page_title="Object Detection App", layout="wide")

st.title("🧠 Object Detection & Counting (YOLO11m Seg)")
st.write("Detect and count selected objects (excluding people).")

# Load model
@st.cache_resource
def load_model():
    model = YOLO("yolo11m-seg.pt")
    model.fuse()  # improves performance
    return model

model = load_model()
class_names = model.names

# Remove "person"
object_options = [name for name in class_names.values() if name != "person"]

# Multi-select
selected_objects = st.multiselect(
    "Select objects to detect:",
    options=object_options,
    default=object_options[:5]
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # 🔥 Resize image (CRITICAL for Streamlit Cloud)
    MAX_SIZE = 640
    h, w = img_array.shape[:2]
    scale = min(MAX_SIZE / max(h, w), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    img_array = cv2.resize(img_array, (new_w, new_h))

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run inference (optimized)
    results = model(img_array, imgsz=640, conf=0.5, device="cpu")[0]

    counts = {}
    annotated_img = img_array.copy()

    # Process detections
    if results.boxes is not None:
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            label = class_names[cls_id]

            # ❌ Skip person
            if label == "person":
                continue

            # ✅ Filter by user selection
            if selected_objects and label not in selected_objects:
                continue

            counts[label] = counts.get(label, 0) + 1

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # Show output
    with col2:
        st.image(annotated_img, caption="Detected Objects", use_column_width=True)

    # Counts
    st.subheader("📊 Object Counts")
    if counts:
        for obj, count in counts.items():
            st.write(f"**{obj}**: {count}")
    else:
        st.write("No selected objects detected.")
