from json import loads
import os
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import credentials, auth
from app.models import User

# Initialize Firebase Admin SDK (ensure the path to your credentials JSON file is correct)
firebase_config = loads(os.environ["Firebase"])
cred = credentials.Certificate(firebase_config)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

security = HTTPBearer()

def verify_token(id_token: str):
    """
    Verify the Firebase ID token sent by the client.
    """
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> User:
    """
    Dependency that verifies the Firebase ID token from the Authorization header.
    Returns a Pydantic User model created from the decoded token.
    """
    token = credentials.credentials
    decoded_token = verify_token(token)
    uid = decoded_token.get("uid")
    email = decoded_token.get("email")
    if not uid or not email:
        raise HTTPException(status_code=400, detail="Token missing required fields")
    return User(firebase_uid=uid, email=email)