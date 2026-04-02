from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.db.models import Avg
from .models import Prediction, Feedback

import os
import pickle
import librosa
import numpy as np


# ================= LOAD MODEL =================
MODEL_PATH = os.path.join(settings.BASE_DIR, "models", "emotion_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model, encoder, scaler = pickle.load(f)


# ================= FEATURE EXTRACTION =================
def extract_features(file_path):

    try:
        y, sr = librosa.load(file_path, sr=22050)

        y, _ = librosa.effects.trim(y)
        y = librosa.util.normalize(y)

        # MFCC (40)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # Delta MFCC (40)
        delta = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta.T, axis=0)

        # Chroma (12)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        # ZCR (1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # RMS (1)
        rms = np.mean(librosa.feature.rms(y=y))

        features = np.hstack((mfcc_mean, delta_mean, chroma_mean, zcr, rms))

        print("Feature shape:", features.shape)  # should show (94,)

        return features.reshape(1, -1)

    except Exception as e:
        print("Feature extraction error:", e)
        return None


# ================= MUSIC DETECTION =================
def detect_music(file_path):

    try:
        y, sr = librosa.load(file_path, sr=22050)

        y, _ = librosa.effects.trim(y)

        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))

        print("Music check -> ZCR:", zcr)
        print("Music check -> Centroid:", centroid)
        print("Music check -> Rolloff:", rolloff)
        print("Music check -> RMS:", rms)

        music_score = 0

        # Music usually has very high centroid
        if centroid > 2900:
            music_score += 1

        # Music spreads across wider frequency
        if rolloff > 5000:
            music_score += 1

        # Music normally has stronger continuous energy
        if rms > 0.07:
            music_score += 1

        print("Music score:", music_score)

        return music_score >= 2

    except Exception as e:
        print("Music detection error:", e)
        return False
        


# ================= SPEECH VALIDATION =================
def is_valid_speech(file_path):

    try:
        y, sr = librosa.load(file_path, sr=22050)

        y, _ = librosa.effects.trim(y)

        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))

        print("Speech ZCR:", zcr)
        print("Speech Centroid:", centroid)
        print("Speech RMS:", rms)

        speech_score = 0

        if 0.04 < zcr < 0.22:
            speech_score += 1

        if centroid < 3500:
            speech_score += 1

        if rms > 0.02:
            speech_score += 1

        print("Speech score:", speech_score)

        return speech_score >= 2

    except Exception as e:
        print("Speech detection error:", e)
        return False


# ================= LANDING PAGE =================
def landing(request):
    return render(request, "emotion_app/home.html")
#silence detector
def is_silent(file_path):

    try:
        y, sr = librosa.load(file_path, sr=22050)

        y, _ = librosa.effects.trim(y)

        rms = np.mean(librosa.feature.rms(y=y))

        print("Silence RMS:", rms)

        # Only treat as silence if energy is extremely low
        if rms < 0.003:
            return True

        return False

    except Exception as e:
        print("Silence detection error:", e)
        return False


# ================= ANALYZE PAGE =================
def analyze(request):

    emotion = None
    confidence = None
    top3 = None
    error_message = None

    history = Prediction.objects.all().order_by('-created_at')[:5]

    if request.method == 'POST' and request.FILES.get('audio_file'):

        audio_file = request.FILES['audio_file']

        try:
            upload_dir = os.path.join(settings.BASE_DIR, "uploads")
            os.makedirs(upload_dir, exist_ok=True)

            file_path = os.path.join(upload_dir, audio_file.name)

            with open(file_path, "wb+") as destination:
                for chunk in audio_file.chunks():
                    destination.write(chunk)

            print("Saved:", file_path)
            # Check silence
            if is_silent(file_path):

                error_message = "No speech detected. Please record your voice."

                return render(
                    request,
                    "emotion_app/analyze.html",
                    {
                        "emotion": None,
                        "confidence": None,
                        "top3": None,
                        "history": history,
                        "error_message": error_message
                    }
                )
             

            # ---------- MUSIC DETECTION ----------
            if detect_music(file_path):

                error_message = "Music detected. Please upload human speech."

            else:

                # ---------- SPEECH VALIDATION ----------
                if not is_valid_speech(file_path):

                    error_message = "No clear human speech detected."

                else:

                    features = extract_features(file_path)

                    if features is None:
                        error_message = "Audio processing failed."

                    else:

                        features = scaler.transform(features)

                        probabilities = model.predict_proba(features)[0]

                        top3_indices = probabilities.argsort()[-3:][::-1]

                        top3 = []

                        for idx in top3_indices:
                            emotion_name = encoder.inverse_transform([idx])[0]
                            conf = round(probabilities[idx] * 100, 2)

                            top3.append({
                                "emotion": emotion_name,
                                "confidence": conf
                            })

                    emotion = top3[0]["emotion"]
                    confidence = top3[0]["confidence"]

                    # AI reliability check
                    if confidence < 30:
                            emotion = "Uncertain Emotion"

                    Prediction.objects.create(
                            emotion=emotion,
                            confidence=confidence
                        )

                    history = Prediction.objects.all().order_by('-created_at')[:5]

        except Exception as e:
            print("Error:", e)
            error_message = "Something went wrong while processing the audio."


    # ---------- FEEDBACK ----------
    if request.method == "POST" and request.POST.get("rating"):

        rating = int(request.POST.get("rating"))
        is_accurate = request.POST.get("is_accurate") == "true"
        comment = request.POST.get("comment")

        Feedback.objects.create(
            prediction=Prediction.objects.last(),
            rating=rating,
            is_accurate=is_accurate,
            comment=comment
        )

        messages.success(request, "Thank you for your feedback!")

        return redirect("analyze")


    return render(
        request,
        "emotion_app/analyze.html",
        {
            "emotion": emotion,
            "confidence": confidence,
            "top3": top3,
            "history": history,
            "error_message": error_message
        }
    )


# ================= DASHBOARD =================
def dashboard(request):

    total_predictions = Prediction.objects.count()

    total_feedback = Feedback.objects.count()

    avg_rating = Feedback.objects.aggregate(Avg("rating"))["rating__avg"]

    accurate_count = Feedback.objects.filter(is_accurate=True).count()

    if total_feedback > 0:
        accuracy_percent = round((accurate_count / total_feedback) * 100, 2)
    else:
        accuracy_percent = 0

    recent_feedback = Feedback.objects.order_by("-id")[:5]

    context = {
        "total_predictions": total_predictions,
        "avg_rating": avg_rating,
        "accuracy_percent": accuracy_percent,
        "recent_feedback": recent_feedback
    }

    return render(request, "emotion_app/dashboard.html", context)


# ================= ABOUT PAGE =================
def about(request):
    return render(request, "emotion_app/about.html")