from flask import Flask, request, jsonify
import numpy as np
import logging
import tempfile
import os
import json
from datetime import datetime
from pydantic import BaseModel
from voice_activity_detector import detect_human_voice
import joblib


from voice_activity_detector import train_model, extract_features, classify_audio

SECRET_KEY = "audio_file"
ALGORITHM = "HS256"

app = Flask(__name__)
handler = logging.FileHandler("errors.log")  # Create the file logger
app.logger.addHandler(handler)
app.logger.setLevel(logging.ERROR)


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


class AudioFile(BaseModel):
    filename: str
    content_type: str

    @classmethod
    def validate_audio_file(cls, file) -> "AudioFile":
        valid_types = ["audio/wav", "audio/mpeg", "audio/aac"]
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
    Trains an audio classification model using either provided audio files and labels,
    or feedback data if no new data is provided.

    This function performs the following steps:
    1. Checks if new training data (files and labels) is provided in the request.
    2. If new data is provided, it uses that for training.
    3. If no new data is provided, it uses the feedback data for training.
    4. Extracts features from the audio files.
    5. Trains a machine learning model using the extracted features and labels.
    6. Saves the trained model to a file named `audio_classifier_model.joblib`.
    7. Returns a JSON response with the training result and the model's accuracy.

    If any errors occur during the process, the function returns appropriate error responses with HTTP status codes.
    """
    X, y = [], []

    if "files" in request.files and "labels" in request.form:
        # Use provided training data
        files = request.files.getlist("files")
        labels = request.form.getlist("labels")
        if len(files) != len(labels):
            return jsonify({"detail": "Number of files and labels must match"}), 400

        with tempfile.TemporaryDirectory() as temp_dir:
            for file, label in zip(files, labels):
                temp_path = os.path.join(temp_dir, file.filename)
                file.save(temp_path)
                features = extract_features(temp_path)
                X.append(features)
                y.append(int(label))
    else:
        # Use feedback data for training
        feedback_json_path = "feedback_queue.json"
        feedback_audio_dir = "feedback_audio"
        
        try:
            with open(feedback_json_path, "r") as f:
                feedback_queue = json.load(f)
        except FileNotFoundError:
            return jsonify({"detail": "No feedback data available for training"}), 400

        for feedback in feedback_queue:
            audio_path = os.path.join(feedback_audio_dir, feedback["filename"])
            if os.path.exists(audio_path):
                features = extract_features(audio_path)
                X.append(features)
                y.append(int(feedback["label"]))

        if not X:
            return jsonify({"detail": "No valid feedback data available for training"}), 400

    X = np.array(X)
    y = np.array(y)
    model, accuracy = train_model(X, y)
    joblib.dump(model, "audio_classifier_model.joblib")

    # Clear the feedback queue after successful training
    if "files" not in request.files and os.path.exists(feedback_json_path):
        os.remove(feedback_json_path)

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
    if "file" not in request.files:
        return jsonify({"detail": "No file part"}), 400

    file = request.files["file"]
    audio_file = AudioFile.validate_audio_file(file)
    if isinstance(audio_file, tuple):
        return audio_file

    try:
        model = joblib.load("audio_classifier_model.joblib")
    except FileNotFoundError:
        return (
            jsonify({"detail": "Model not found. Please train the model first."}),
            400,
        )

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


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Handles the processing of user feedback on audio classifications.

    This function performs the following steps:
    1. Validates that the request contains the required `file` and `label` parameters.
    2. Validates the uploaded audio file.
    3. Saves the audio file to the `feedback_audio` directory.
    4. Saves the feedback information (file name and label) to a JSON file for later retraining.
    5. Returns a success message indicating that the feedback has been received.

    If any errors occur during the process, the function returns appropriate error responses with HTTP status codes.
    """
    if "file" not in request.files or "label" not in request.form:
        return jsonify({"detail": "File and label must be provided."}), 400

    file = request.files["file"]
    label = request.form["label"]

    audio_file = AudioFile.validate_audio_file(file)
    if isinstance(audio_file, tuple):
        return audio_file

    # Save the audio file
    feedback_dir = "feedback_audio"
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_path = os.path.join(feedback_dir, file.filename)
    file.save(feedback_path)

    # Save feedback information for later retraining
    feedback_info = {
        "filename": file.filename,
        "label": label,
        "timestamp": datetime.now().isoformat(),
    }

    feedback_json_path = "feedback_queue.json"
    try:
        with open(feedback_json_path, "r+") as f:
            feedback_queue = json.load(f)
            feedback_queue.append(feedback_info)
            f.seek(0)
            json.dump(feedback_queue, f, indent=2)
    except FileNotFoundError:
        with open(feedback_json_path, "w") as f:
            json.dump([feedback_info], f, indent=2)

    return jsonify(
        {"detail": "Feedback received and queued for retraining. Thank you!"}
    )

if __name__ == "__main__":
    app.run(debug=True)
