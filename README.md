# 🫁 Lung Disease Detection (Chest X-rays)

This project uses **Convolutional Neural Networks (CNNs)** to detect lung diseases (Normal vs Pneumonia) from chest X-ray images.  
It also provides an interactive **Streamlit UI** to upload X-rays and see predictions + Grad-CAM heatmaps.

---

## 📂 Project Structure

lung_disease_detection/
│── data/                         # Dataset (Chest X-rays/CT scans)
│   ├── train/                    # Training images
│   ├── val/                      # Validation images
│   ├── test/                     # Test images
│
│── models/                       # Saved ML models
│   └── lung_model.h5             # Trained CNN model
│
│── notebooks/                    
│   └── model_training.ipynb      # Jupyter notebook for experiments
│
│── src/
│   ├── preprocessing.py          # Image preprocessing functions
│   ├── train_model.py            # Training pipeline
│   ├── predict.py                # Prediction & Grad-CAM
│
│── app.py                        # Streamlit UI
│── requirements.txt              # Dependencies
│── README.md                     # Project documentation
