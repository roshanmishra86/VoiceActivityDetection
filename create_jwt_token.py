from datetime import datetime, timedelta
from jose import jwt

SECRET_KEY = "audio_file"
ALGORITHM = "HS256"


def create_jwt_token():
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {"sub": "user_id", "exp": expiration}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


# Generate and print the token
print(create_jwt_token())
