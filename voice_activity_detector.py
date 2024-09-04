import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, res_type="kaiser_fast")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    features = np.hstack(
        [
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1),
        ]
    )

    return features


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


def classify_audio(model, audio_path):
    features = extract_features(audio_path)
    prediction = model.predict([features])[0]

    if prediction == 0:
        return "Blank (No sound)"
    elif prediction == 1:
        return "Background noise only"
    else:
        return "Human voice with background"
