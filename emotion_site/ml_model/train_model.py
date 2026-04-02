import os
import librosa
import numpy as np
import pickle

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAVDESS_PATH = os.path.join(BASE_DIR, "dataset", "RAVDESS")
CREMA_PATH = os.path.join(BASE_DIR, "dataset", "CREMA-D")
TESS_PATH = os.path.join(BASE_DIR, "dataset", "TESS")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ================= RAVDESS EMOTION MAP =================
ravdess_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised"
}


# ================= CREMA EMOTION MAP =================
crema_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}


# ================= FEATURE EXTRACTION =================
def extract_features(file_path):

    y, sr = librosa.load(file_path, duration=3)

    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Delta MFCC
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta.T, axis=0)

    # Chroma (NEW - improves accuracy)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # RMS
    rms = np.mean(librosa.feature.rms(y=y))

    return np.hstack((mfcc_mean, delta_mean, chroma_mean, zcr, rms))


X = []
y = []

# ================= LOAD RAVDESS =================
print("Reading RAVDESS...")

for actor in os.listdir(RAVDESS_PATH):

    actor_path = os.path.join(RAVDESS_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        parts = file.split("-")

        # Skip invalid file names
        if len(parts) < 3:
            continue

        emotion_code = parts[2]

        if emotion_code not in ravdess_map:
            continue

        emotion = ravdess_map[emotion_code]

        file_path = os.path.join(actor_path, file)

        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(emotion.lower())
        except:
            continue

print("Reading CREMA-D...")

for file in os.listdir(CREMA_PATH):

    if not file.endswith(".wav"):
        continue

    parts = file.split("_")

    if len(parts) < 3:
        continue

    emotion_code = parts[2]

    emotion = crema_map.get(emotion_code)

    if emotion:

        file_path = os.path.join(CREMA_PATH, file)

        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(emotion.lower())
        except:
            continue

# ================= LOAD TESS =================
print("Reading TESS...")

for emotion_folder in os.listdir(TESS_PATH):

    folder_path = os.path.join(TESS_PATH, emotion_folder)

    emotion = emotion_folder.split("_")[-1]

    for file in os.listdir(folder_path):

        if file.endswith(".wav"):

            file_path = os.path.join(folder_path, file)

            features = extract_features(file_path)

            X.append(features)
            y.append(emotion.lower())


X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))


# ================= ENCODE =================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# ================= SCALE =================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ================= TRAIN SVM =================
model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True
)

print("Training model...")

model.fit(X_train, y_train)


# ================= EVALUATE =================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred, target_names=encoder.classes_))


# ================= SAVE MODEL =================
with open(MODEL_PATH, "wb") as f:

    pickle.dump((model, encoder, scaler), f)


print("\nModel saved successfully!")