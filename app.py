import streamlit as st
from src.predict import predict_disease
from PIL import Image
import time
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Lung Disease Detection", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Lung Disease Detection from Chest X-rays</h1>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Instructions
# -----------------------------
with st.expander("ℹ️ How to use this app"):
    st.write("""
    1. Upload a chest X-ray image in JPG, JPEG, or PNG format.  
    2. Wait a few seconds while the model predicts.  
    3. View the prediction label, confidence, and inference time.  
    4. Compare with previous predictions if available.
    """)

# -----------------------------
# Initialize session state
# -----------------------------
if "prev_prediction" not in st.session_state:
    st.session_state.prev_prediction = None
if "prev_confidence" not in st.session_state:
    st.session_state.prev_confidence = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# -----------------------------
# Upload section
# -----------------------------
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file is not None:
    uploaded_file = st.session_state.uploaded_file

    # Display uploaded image in left column
    col1, col2 = st.columns([1, 1])
    
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-ray", use_container_width=True)
    
    # Save temporarily
    temp_path = "temp_image.jpg"
    img.save(temp_path)
    
    # -----------------------------
    # Predict with simulated loading
    # -----------------------------
    with col2:
        st.info("🔎 Predicting...")
        start_time = time.time()
        progress_bar = st.progress(0)
        for i in range(1, 101):
            time.sleep(0.01)
            progress_bar.progress(i)
        
        label, confidence, _ = predict_disease(temp_path)
        inference_time = time.time() - start_time
        st.success("✅ Prediction Complete!")

        # -----------------------------
        # Color-coded prediction
        # -----------------------------
        label_color = "green" if label == "NORMAL" else "red"
        st.markdown(
            f"<h3>Prediction: <span style='color:{label_color}'>{label}</span></h3>",
            unsafe_allow_html=True
        )

        # -----------------------------
        # Colored confidence bar
        # -----------------------------
        conf_percentage = int(confidence * 100)
        if conf_percentage >= 70:
            bar_color = "#FF4B4B" if label == "PNEUMONIA" else "#4CAF50"
        else:
            bar_color = "#FFC107"  # amber for lower confidence

        st.markdown(f"""
        <div style="background-color: #ddd; border-radius: 10px; width: 100%; height: 25px;">
            <div style="width: {conf_percentage}%; background-color: {bar_color}; height: 100%; border-radius: 10px;"></div>
        </div>
        <p style='text-align:center'>{conf_percentage}% confidence</p>
        """, unsafe_allow_html=True)

        # -----------------------------
        # Show inference time
        # -----------------------------
        st.info(f"Inference time: {inference_time:.2f} seconds")

        # -----------------------------
        # Compare with previous prediction
        # -----------------------------
        if st.session_state.prev_prediction:
            st.markdown("### 🔄 Comparison with previous prediction")
            st.markdown(f"**Previous Prediction:** {st.session_state.prev_prediction} ({st.session_state.prev_confidence*100:.2f}% confidence)")
            st.markdown(f"**Current Prediction:** {label} ({conf_percentage}%)")
        
        # Update session state
        st.session_state.prev_prediction = label
        st.session_state.prev_confidence = confidence

    st.markdown("---")
    st.markdown("Made with ❤️ using TensorFlow and Streamlit")
