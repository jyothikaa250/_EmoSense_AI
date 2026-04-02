# 🎤 EmoSense AI – Voice-Based Emotion Detection System

EmoSense AI is a machine learning-based web application that detects human emotions from voice input. The system analyzes speech signals and predicts emotions in real-time using advanced audio processing techniques.

---

## 🚀 Features

- 🎤 Real-time voice recording
- 📁 Audio file upload for prediction
- 🧠 Machine Learning (SVM) based emotion classification
- 🎵 Music detection (filters non-speech audio)
- 🔇 Silence detection for validation
- 📊 Top-3 emotion predictions with confidence scores
- ⭐ User feedback system
- 📈 Admin dashboard for monitoring

---

## 🛠️ Tech Stack

- Python
- Django
- Scikit-learn
- Librosa
- HTML, CSS, JavaScript

---

## ⚙️ How It Works

Audio Input → Preprocessing → Validation (Silence & Music Detection) → Feature Extraction → Emotion Prediction → Result Display

---

## ▶️ How to Run

```bash
cd emotion_site
python manage.py runserver
