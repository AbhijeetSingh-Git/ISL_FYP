# 🤟 Indian Sign Language (ISL) Recognition System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.8-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red)

A comprehensive, end-to-end Final Year Project for recognizing Indian Sign Language (ISL) from video inputs. The system translates continuous sign language gestures into text and synthesizes that text into audio/speech.

## 🌟 Key Features

* **Video-to-Text Translation**: Upload a video of a person performing sign language to receive the predicted text word/sentence.
* **Text-to-Speech (TTS)**: The predicted text is converted into an audio format using Google Text-to-Speech (gTTS).
* **Deep Learning Model**: Utilizes an LSTM (Long Short-Term Memory) neural network to understand sequential temporal patterns in sign gestures.
* **Advanced Feature Extraction**: Employs Google's MediaPipe framework to accurately extract holistic keypoints (body, face, and hands) from video frames.
* **Interactive UI**: Built with Streamlit, providing a premium, user-friendly, and modern web interface.

## 🛠️ Technology Stack

* **Programming Language**: Python
* **Computer Vision**: OpenCV, MediaPipe
* **Deep Learning Framework**: TensorFlow / Keras (LSTM architecture)
* **Data Processing**: NumPy, Scikit-learn
* **Web UI Framework**: Streamlit
* **Text-to-Speech**: gTTS

## 📂 Project Structure

```text
Indian-Sign-Language-Recognition-System/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── models/                     # Saved model artifacts (weights, encoder, stats)
│   ├── sign_language_model.h5
│   ├── label_encoder.npy
│   └── norm_stats.npz
├── src/                        # Source code for training & preprocessing
│   ├── data_preprocessing.py   # Code for extracting frames & sequence handling
│   ├── feature_extraction.py   # MediaPipe keypoint extraction logic
│   ├── model.py                # Definition of the LSTM model architecture
│   ├── train.py                # Script to execute the model training pipeline
│   └── predict.py              # Prediction and Text-to-Speech helper modules
└── videos/                     # Directory for dataset or test video files
```

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python installed (preferably version 3.8 to 3.11). Create an isolated virtual environment to avoid dependency conflicts:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installation

Install all required Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Training the Model (Optional)

If you haven't trained the model yet, or if you want to retrain it on your dataset:

```bash
python src/train.py
```
*This will generate `sign_language_model.h5`, `label_encoder.npy`, and `norm_stats.npz` in the `models/` directory.*

### 4. Running the Web Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

The application will open in your default browser automatically (usually at `http://localhost:8501`).

## 💡 Usage

1. Open the Streamlit web interface.
2. Click on **"Choose a sign language video"** and upload a `.mp4`, `.avi`, or `.mov` file.
3. Click the **"🔍 Recognize Sign Language"** button.
4. The system will process the video, extract keypoints, and predict the sign.
5. The predicted text, model confidence score, and generated audio reading the prediction will appear on the screen!

## 🎓 About

This repository serves as a major project reflecting the process of building an AI pipeline—from dataset collection and feature engineering to model training and interactive final deployment.
