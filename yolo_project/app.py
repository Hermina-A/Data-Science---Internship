import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IntelliVision AI | Object Detection System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHE YOLO MODEL ---
@st.cache_resource
def load_model():
    # Loads nano model by default. Replace with your custom 'best.pt' if needed.
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

# --- SESSION STATE MANAGEMENT ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "users" not in st.session_state:
    st.session_state.users = {"admin": "admin123"}  # Default mock user

# --- NAVIGATION HELPER ---
def nav_to(page):
    st.session_state.current_page = page

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🤖 IntelliVision AI")
    st.subheader("Next-Gen Object Detection")
    st.write("---")
    
    if st.session_state.logged_in:
        st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            nav_to("Home")
            st.rerun()
        
        st.write("---")
        # App internal pages
        if st.button("🏠 Home Dashboard", use_container_width=True): nav_to("Home")
        if st.button("📷 Real-time Inference", use_container_width=True): nav_to("Detection App")
        if st.button("ℹ️ About the Project", use_container_width=True): nav_to("About")
    else:
        st.sidebar.warning("Please Login or Register to access the full system.")
        if st.button("🔐 Login", use_container_width=True): nav_to("Login")
        if st.button("📝 Register", use_container_width=True): nav_to("Register")
        if st.button("ℹ️ About", use_container_width=True): nav_to("About")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 1rem; }
    .subtitle { font-size: 1.2rem; color: #4B5563; margin-bottom: 2rem; }
    .card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)


# ==========================================
# PAGE 1: LOGIN
# ==========================================
if st.session_state.current_page == "Login":
    st.markdown("<div class='main-title'>Account Login</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if username in st.session_state.users and st.session_state.users[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Login Successful!")
                    time.sleep(0.5)
                    nav_to("Home")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password.")
    with col2:
        st.info("💡 **Demo Credentials**:\n\nUsername: `admin` \n\nPassword: `admin123`")


# ==========================================
# PAGE 2: REGISTER
# ==========================================
elif st.session_state.current_page == "Register":
    st.markdown("<div class='main-title'>Create an Account</div>", unsafe_allow_html=True)
    
    with st.form("register_form"):
        new_user = st.text_input("Choose a Username")
        new_pass = st.text_input("Choose a Password", type="password")
        confirm_pass = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register", use_container_width=True)
        
        if submitted:
            if not new_user or not new_pass:
                st.error("Fields cannot be empty.")
            elif new_user in st.session_state.users:
                st.error("Username already exists.")
            elif new_pass != confirm_pass:
                st.error("Passwords do not match.")
            else:
                st.session_state.users[new_user] = new_pass
                st.success("Registration Successful! You can now log in.")
                time.sleep(0.5)
                nav_to("Login")
                st.rerun()


# ==========================================
# PAGE 3: HOME DASHBOARD
# ==========================================
elif st.session_state.current_page == "Home":
    st.markdown("<div class='main-title'>Welcome to IntelliVision AI Hub</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>A unified intelligence engine tailored for real-time edge detection and custom use cases.</div>", unsafe_allow_html=True)
    
    # Showcase Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='card'><h3>🦺 Safety & PPE</h3><p>Detect vests, helmets, goggles, and compliance risks instantly on site.</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='card'><h3>🚗 Traffic Analytics</h3><p>Monitor bottlenecks, detect vehicle types, and capture lane violations seamlessly.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='card'><h3>🛡️ Smart Surveillance</h3><p>Intruder detection, weapon classification, and perimeter protection modules.</p></div>""", unsafe_allow_html=True)

    st.write("---")
    
    if not st.session_state.logged_in:
        st.info("🔒 Please log in from the sidebar to unlock the **Real-time Inference** engine.")
    else:
        st.success(f"⚡ System Fully Armed. Welcome back, {st.session_state.username}! Click 'Real-time Inference' on the sidebar to begin detections.")


# ==========================================
# PAGE 4: ABOUT
# ==========================================
elif st.session_state.current_page == "About":
    st.markdown("<div class='main-title'>About the Project</div>", unsafe_allow_html=True)
    st.write("""
    This Object Detection Framework leverages **State-of-the-Art Deep Learning Architecture (YOLO)** paired with high-performance visualization analytics. 
    Designed for scalability, it serves critical industrial frameworks spanning infrastructure safety, automated traffic policing, and intelligent surveillance ecosystems.
    """)
    
    st.subheader("Key Architecture Features")
    st.markdown("""
    - **YOLOv8 Engine Architecture**: Multi-scale feature maps allowing ultra-fast object bounding box and confidence estimations.
    - **Interactive Visual Analytics**: Dynamic charts monitoring object density matrices and frame-by-frame analysis tracking counts.
    - **Flexible Modality Framework**: Accepts static standard resolution images, video arrays, and real-time live local webcam capture feeds.
    """)
    
    st.info("Developed utilizing Python, Streamlit, OpenCV, and Ultralytics Deep Learning Modules.")


# ==========================================
# PAGE 5: CORE DETECTION APP (AUTHENTICATED)
# ==========================================
elif st.session_state.current_page == "Detection App":
    if not st.session_state.logged_in:
        st.warning("Unauthorized Access. Redirecting to Home.")
        nav_to("Home")
        st.rerun()

    st.markdown("<div class='main-title'>Computer Vision Inference Studio</div>", unsafe_allow_html=True)
    
    # Detection Settings
    with st.expander("⚙️ Adjust Detection Confidence Thresholds", expanded=False):
        conf_threshold = st.slider("Model Confidence Threshold", min_value=0.1, max_value=1.0, value=0.4, step=0.05)

    # Input Modality Selection
    mode = st.radio("Choose Input Modality Source:", ["📸 Image Upload", "🎥 Video Upload", "📹 Live Webcam Video"], horizontal=True)
    st.write("---")

    # ------------------- MODE A: IMAGE UPLOAD -------------------
    if mode == "📸 Image Upload":
        uploaded_file = st.file_uploader("Upload Target Image File...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Perform Inference
            with st.spinner("Processing Frame via YOLO Engine..."):
                results = model.predict(source=img_array, conf=conf_threshold)
                
                # Render results on top of image
                res_plotted = results[0].plot()
                
                # Generate Analytics Meta
                boxes = results[0].boxes
                detected_classes = [model.names[int(box.cls[0])] for box in boxes]
                confidences = [float(box.conf[0]) for box in boxes]

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Inference Result Output")
                st.image(res_plotted, channels="BGR", use_container_width=True)
            
            with col2:
                st.subheader("Visual Analytics Metrics")
                if len(detected_classes) > 0:
                    df = pd.DataFrame({"Object": detected_classes, "Confidence": confidences})
                    
                    # Count Metric Table
                    count_df = df["Object"].value_counts().reset_index()
                    count_df.columns = ["Detected Object", "Total Instances Count"]
                    st.dataframe(count_df, use_container_width=True)
                    
                    # Plotly chart
                    fig = px.bar(count_df, x="Detected Object", y="Total Instances Count", 
                                 title="Detected Target Object Densities", color="Detected Object")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No targeted objects met the designated confidence threshold matrix.")

    # ------------------- MODE B: VIDEO UPLOAD -------------------
    elif mode == "🎥 Video Upload":
        uploaded_video = st.file_uploader("Upload Target Video File...", type=["mp4", "avi", "mov"])
        
        if uploaded_video is not None:
            # Save uploaded stream transiently to disk
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_video.read())
            
            vid_cap = cv2.VideoCapture("temp_video.mp4")
            st_frame = st.empty()
            analytics_area = st.empty()
            
            stop_btn = st.button("Stop Video Processing Loop")
            
            while vid_cap.isOpened():
                ret, frame = vid_cap.read()
                if not ret or stop_btn:
                    break
                
                # Inference
                results = model.predict(source=frame, conf=conf_threshold, verbose=False)
                res_plotted = results[0].plot()
                
                # Update Video Element Frame
                st_frame.image(res_plotted, channels="BGR", use_container_width=True)
                
                # Live Analytics Update
                boxes = results[0].boxes
                detected_classes = [model.names[int(box.cls[0])] for box in boxes]
                if len(detected_classes) > 0:
                    df = pd.DataFrame({"Object": detected_classes})
                    count_df = df["Object"].value_counts().reset_index()
                    count_df.columns = ["Class", "Count"]
                    analytics_area.write(f"**Live Tracking State Matrix:** " + ", ".join([f"{row['Class']}: {row['Count']}" for _, row in count_df.iterrows()]))
                    
            vid_cap.release()
            st.success("Video Analytics Routine Completed.")

    # ------------------- MODE C: LIVE WEBCAM -------------------
    elif mode == "📹 Live Webcam Video":
        st.write("⚠️ *Note: Ensure browser permissions allow local camera hardware mapping.*")
        run_cam = st.toggle("Power Active Video Capture System Stream")
        
        if run_cam:
            # Connect to local index 0 device camera array
            vid_cap = cv2.VideoCapture(0)
            st_frame = st.empty()
            analytics_area = st.empty()
            
            while vid_cap.isOpened() and run_cam:
                ret, frame = vid_cap.read()
                if not ret:
                    st.error("Failed to interface with capture array device hardware.")
                    break
                
                # Inference
                results = model.predict(source=frame, conf=conf_threshold, verbose=False)
                res_plotted = results[0].plot()
                
                # Stream frames onto client window
                st_frame.image(res_plotted, channels="BGR", use_container_width=True)
                
                # Mini dynamic stats
                boxes = results[0].boxes
                detected_classes = [model.names[int(box.cls[0])] for box in boxes]
                if len(detected_classes) > 0:
                    df = pd.DataFrame({"Object": detected_classes})
                    count_df = df["Object"].value_counts().reset_index()
                    count_df.columns = ["Class", "Count"]
                    analytics_area.write(f"**Active Live Telemetry Tracks:** " + ", ".join([f"{row['Class']}: {row['Count']}" for _, row in count_df.iterrows()]))
                else:
                    analytics_area.write("Scanning environment... No active signatures localized.")
                    
            vid_cap.release()
            st.write("Video Stream Pipeline Rested.")