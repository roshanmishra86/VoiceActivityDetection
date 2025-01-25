# Voice Activity Detection API 🎤

A Flask-based REST API that detects human voice activity in audio files. Perfect for those moments when you need to know if someone's actually speaking or if it's just your neighbour's cat knocking things over.

## Features

- Audio classification into three categories:
  - Blank (No sound)
  - Background noise only
  - Human voice with background
- Continuous learning through user feedback
- JWT-based authentication
- Support for WAV, MP3, and AAC audio formats
- Noise reduction preprocessing
- Feature extraction using MFCCs, spectral centroids, and chroma features

## Technical Architecture

The system follows a modular architecture with these core components:

- **Feature Extraction**: Uses librosa to extract meaningful audio features (MFCCs, spectral centroids, chroma)
- **Model**: RandomForest classifier trained on the extracted features
- **API**: Flask-based REST endpoints for training, classification, and feedback
- **Authentication**: JWT-based token system for secure access

## Prerequisites

```
Python 3.8+
FFmpeg (for audio processing)
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Setup

Set your JWT secret key in `create_jwt_token.py`:
```python
SECRET_KEY = "your_secure_secret_key"
```

## API Endpoints

### Training Endpoint
`POST /train`
- Train the model with new audio files and labels
- Can also use accumulated feedback data if no new files are provided

### Classification Endpoint
`POST /upload-audio`
- Upload an audio file for classification
- Returns classification result and confidence score

### Feedback Endpoint
`POST /feedback`
- Submit feedback for improving model accuracy
- Stores audio file and correct label for future retraining

## Request Examples

Training with new files:
```bash
curl -X POST \
  http://localhost:5000/train \
  -H 'Authorization: Bearer YOUR_JWT_TOKEN' \
  -F 'files=@audio1.mp3' \
  -F 'files=@audio2.wav' \
  -F 'labels=1' \
  -F 'labels=2'
```

Classifying an audio file:
```bash
curl -X POST \
  http://localhost:5000/upload-audio \
  -H 'Authorization: Bearer YOUR_JWT_TOKEN' \
  -F 'file=@audio.mp3'
```

## Model Details

The system uses a Random Forest Classifier with:
- 100 estimators
- Feature set combining MFCCs, spectral centroids, and chroma features
- Noise reduction preprocessing using the noisereduce library

## Project Structure
```
├── main.py                 # Flask application and API endpoints
├── voice_activity_detector.py  # Core ML functionality
├── create_jwt_token.py     # JWT token generation
├── requirements.txt        # Project dependencies
└── feedback_audio/        # Directory for feedback audio files
```

## Error Handling

The API implements comprehensive error handling for:
- Invalid file types
- Missing JWT tokens
- Model not found scenarios
- Invalid feedback formats
- File processing errors

Errors are logged to `errors.log` for debugging and monitoring.

## Best Practices

1. Always validate audio files before processing
2. Use error handling for robust production deployment
3. Regularly retrain the model with feedback data
4. Monitor the feedback queue size
5. Back up the trained model regularly

## Future Improvements

1. Add model versioning
2. Implement batch processing for large audio files
3. Add confidence scores to classifications
4. Implement real-time streaming classification
5. Add more granular voice activity categories

## Security Notes

1. Change the default SECRET_KEY
2. Implement rate limiting
3. Add request size limitations
4. Implement proper file cleanup
5. Consider adding API key rotation

## Contributing

Feel free to open issues and pull requests. Just make sure your code doesn't classify heavy metal as "background noise" - we've been there, it wasn't pretty.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Note: This software depends on other packages that may be licensed under different terms.
The MIT license above applies only to the original code in this repository.
Notably, this project uses FFmpeg which is licensed under the LGPL/GPL license.
