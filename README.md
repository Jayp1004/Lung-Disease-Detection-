# 🫁 Lung Disease Detection (Chest X-rays)

This project uses **Convolutional Neural Networks (CNNs)** to detect lung diseases (Normal vs Pneumonia) from chest X-ray images.  
It also provides an interactive **Streamlit UI** to upload X-rays and see predictions + Grad-CAM heatmaps.

---

## 📂 Project Structure

lung_disease_detection/
│── data/ # Dataset (Chest X-rays)
│ ├── train/
│ ├── val/
│ ├── test/
│
│── models/
│ └── lung_model.h5 # Saved CNN model (after training)
│
│── notebooks/
│ └── model_training.ipynb # Jupyter notebook for experiments
│
│── src/
│ ├── preprocessing.py # Image preprocessing functions
│ ├── train_model.py # Training pipeline
│ ├── predict.py # Prediction & Grad-CAM
│
│── app.py # Streamlit frontend
│── requirements.txt # Dependencies
│── README.md # Documentation