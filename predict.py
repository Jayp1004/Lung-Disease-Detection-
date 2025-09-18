import os
import numpy as np
import cv2
import tensorflow as tf

# -----------------------------
# Paths and constants
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lung_model.h5")
IMG_SIZE = (150, 150)

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
model.trainable = False  # Ensure no accidental training

# -----------------------------
# Preprocess image
# -----------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img_array = np.expand_dims(img / 255.0, axis=0)
    return img_array, img

# -----------------------------
# Predict disease
# -----------------------------
def predict_disease(img_path):
    img_array, img = preprocess_image(img_path)
    pred = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, confidence, img
