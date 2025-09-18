*🩺 Lung Disease Detection from Chest X-rays*

An AI-powered web application built with TensorFlow and Streamlit to classify chest X-ray images as Normal or Pneumonia.

This project helps demonstrate how deep learning models can be applied in the medical imaging domain to assist in disease detection.



🚀 Features

📂 Upload chest X-ray images (JPG, JPEG, PNG)

🔎 Real-time prediction using a trained CNN/ResNet model

🎨 Color-coded predictions:

🟢 Green → Normal

🔴 Red → Pneumonia

📊 Confidence bar with percentage

⏱️ Shows inference time per prediction

🔄 Compares with previous predictions

📱 Mobile-friendly, interactive UI



🛠️ Tech Stack

Python 3.9+

TensorFlow / Keras – Model training & inference

OpenCV – Image preprocessing

Streamlit – Web application

PIL (Pillow) – Image handling



⚙️ Installation

1.Clone the repository

git clone https://github.com/your-username/lung-disease-detection.git

cd lung-disease-detection

2.Create and activate virtual environment

python3 -m venv venv

source venv/bin/activate   # Mac/Linux

venv\Scripts\activate      # Windows

3.Install dependencies

pip install -r requirements.txt

4.Download Dataset

Use the Chest X-ray dataset (Kaggle link)

Place it inside the data/chest_xray/ folder (with train, val, test subfolders).

🏋️ Train Model

Run the training script:

python src/train_model.py

The trained model will be saved inside src/models/lung_model.h5.

🔮 Run the App

Start the Streamlit app:

streamlit run app.py

Then open your browser at http://localhost:8501.

⚠️Disclaimer

This project is only for education purposes.

It is not for a medical diagnostic tool and should not be used as a subsitute for professional healthcare.
