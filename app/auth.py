from __future__ import annotations

import os
from json import loads

import firebase_admin
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth, credentials

from app.db import api_key_check
from app.models import User

# Initialize Firebase Admin SDK (ensure the path to your credentials JSON file is correct)
firebase_config = loads(os.environ["Firebase"])
cred = credentials.Certificate(firebase_config)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

security = HTTPBearer(auto_error=False)
X_API_KEY = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_token(id_token: str) -> dict:
    """Verify the Firebase ID token sent by the client."""
    try:
        return auth.verify_id_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from e


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> User:
    """Dependency that verifies the Firebase ID token from the Authorization header.

    Returns a Pydantic User model created from the decoded token.
    """
    token = credentials.credentials
    decoded_token = verify_token(token)
    uid = decoded_token.get("uid")
    email = decoded_token.get("email")
    if not uid or not email:
        raise HTTPException(status_code=400, detail="Token missing required fields")
    return User(firebase_uid=uid, email=email)


async def authenticate(
    x_api_key: str | None = Security(X_API_KEY),
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> User:
    if x_api_key:
        if not api_key_check(x_api_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid API Key. Check that you are passing a 'X-API-Key' on your header.",
            )
        return User(firebase_uid="apiKeyUser", email="apiKeyUser@openprobono.com")

    if credentials:
        return await get_current_user(credentials)
    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials. Check that you are passing a 'X-API-Key' header or a Firebase ID token.",
    )
