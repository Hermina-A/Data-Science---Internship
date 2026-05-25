import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the YOLOv8 model
@st.cache_resource  # Caches the model so it doesn't reload on every click
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Dictionary mapping friendly names to YOLO class IDs
CLASS_OPTIONS = {
    "Person": 0,
    "Bicycle": 1,
    "Car": 2,
    "Motorcycle": 3,
    "Cat": 15,
    "Dog": 16,
    "Backpack": 24,
    "Umbrella": 25,
    "Handbag": 26,
    "Tie": 27,
    "Suitcase": 28,
    "Cell Phone": 67
}

# --- Front End Layout ---
st.title("🎯 YOLOv8 Single Object Detector")
st.write("Upload an image and choose which specific object you want to detect.")

# Sidebar controls
st.sidebar.header("Settings")
selected_label = st.sidebar.selectbox("Select object to detect:", list(CLASS_OPTIONS.keys()))
target_class_id = CLASS_OPTIONS[selected_label]

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image PIL can read
    image = Image.open(uploaded_file)
    
    # Setup columns to show Before and After side-by-side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    # Run object detection when the user clicks the button
    if st.button(f"Detect Only {selected_label}s"):
        with st.spinner("Processing image..."):
            # Convert PIL image to numpy array format for OpenCV/YOLO
            img_array = np.array(image)
            
            # Run YOLO tracking filtered by the selected class ID
            results = model(img_array, classes=[target_class_id])
            
            # Plot the bounding boxes onto the frame
            annotated_img = results[0].plot()
            
            # Convert BGR back to RGB for streamlit display accuracy
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Detection Result")
                st.image(annotated_img_rgb, use_container_width=True)
                st.success(f"Done! Displaying only detected {selected_label}s.")