from flask import Flask, request, jsonify
import numpy as np
import logging
import tempfile
import os
from pydantic import BaseModel
from jose import JWTError, jwt
from voice_activity_detector import detect_human_voice
import joblib


from voice_activity_detector import train_model, extract_features, classify_audio

SECRET_KEY = "audio_file"
ALGORITHM = "HS256"

app = Flask(__name__)
handler = logging.FileHandler("errors.log")  # Create the file logger
app.logger.addHandler(handler)
app.logger.setLevel(logging.ERROR)


def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return jsonify({"detail": "Invalid token"}), 401


class AudioFile(BaseModel):
    filename: str
    content_type: str

    @classmethod
    def validate_audio_file(cls, file) -> "AudioFile":
        """
        Detects Audio file
        """
        valid_types = [
            "audio/wav",
            "audio/mpeg",
            "audio/aac",
            "application/octet-stream",
        ]
        if file.content_type not in valid_types:
            return (
                jsonify(
                    {"detail": "Invalid file type. Only wav, mp3, and aac are allowed."}
                ),
                400,
            )
        return cls(filename=file.filename, content_type=file.content_type)


@app.route("/train", methods=["POST"])
def train():
    """
    Trains an audio classification model using the provided audio files and labels.

    This function expects the following request parameters:
    - `Authorization` header: A valid JWT token for authentication.
    - `files`: A list of audio files to use for training.
    - `labels`: A list of labels (integers) corresponding to the audio files.

    The function performs the following steps:
    1. Validates the JWT token in the `Authorization` header.
    2. Checks that the `files` and `labels` parameters are present and have the same length.
    3. Extracts audio features from the provided audio files.
    4. Trains a machine learning model using the extracted features and labels.
    5. Saves the trained model to a file named `audio_classifier_model.joblib`.
    6. Returns a JSON response with the detail "Model trained successfully" and the model's accuracy.

    If any errors occur during the process, the function returns appropriate error responses with HTTP status codes.
    """
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"detail": "Authorization header missing"}), 401

    token = token.split(" ")[1]  # Remove "Bearer" prefix
    decode_result = decode_jwt(token)
    if isinstance(decode_result, tuple):
        return decode_result  # Return error response if JWT is invalid

    if "files" not in request.files:
        return jsonify({"detail": "No file part"}), 400

    files = request.files.getlist("files")
    labels = request.form.getlist("labels")

    if len(files) != len(labels):
        return jsonify({"detail": "Number of files and labels must match"}), 400

    X = []
    y = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file, label in zip(files, labels):
            temp_path = os.path.join(temp_dir, file.filename)
            file.save(temp_path)
            features = extract_features(temp_path)
            X.append(features)
            y.append(int(label))

    X = np.array(X)
    y = np.array(y)

    model, accuracy = train_model(X, y)

    # Save the model
    joblib.dump(model, "audio_classifier_model.joblib")
    # pickle and save the trained model

    return jsonify({"detail": "Model trained successfully", "accuracy": accuracy})


@app.route("/upload-audio", methods=["POST"])
def upload_audio():
    """
    Handles the upload and classification of audio files.

    This function performs the following steps:
    1. Validates the JWT token in the `Authorization` header.
    2. Checks that the `file` parameter is present in the request.
    3. Validates the uploaded audio file.
    4. Loads the pre-trained audio classification model.
    5. Classifies the uploaded audio file using the loaded model.
    6. Returns a JSON response with the file name, whether the audio contains human voice, and the classification result.

    If any errors occur during the process, the function returns appropriate error responses with HTTP status codes.
    """
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"detail": "Authorization header missing"}), 401

    token = token.split(" ")[1]  # Remove "Bearer" prefix
    decode_result = decode_jwt(token)
    if isinstance(decode_result, tuple):
        return decode_result  # Return error response if JWT is invalid

    if "file" not in request.files:
        return jsonify({"detail": "No file part"}), 400

    file = request.files["file"]
    audio_file = AudioFile.validate_audio_file(file)
    if isinstance(audio_file, tuple):
        return audio_file  # Return error response if file validation fails

    # Load the trained model
    try:
        model = joblib.load("audio_classifier_model.joblib")
    except FileNotFoundError:
        return (
            jsonify({"detail": "Model not found. Please train the model first."}),
            400,
        )

    # Classify the audio
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file.save(temp_file.name)
        result = classify_audio(model, temp_file.name)

    has_human_voice = result == "Human voice with background"

    return jsonify(
        {
            "file_name": file.filename,
            "has_human_voice": has_human_voice,
            "classification": result,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
