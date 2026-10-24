import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from src.shared.config import settings
from src.shared.database import get_db
from src.shared.models import IoTDevice, User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


# ============================================================
# USER PASSWORD SECURITY
# ============================================================

def hash_password(password: str) -> str:
    """Hash plaintext password with bcrypt."""
    pw_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plaintext password against bcrypt hash."""
    pw_bytes = plain_password.encode("utf-8")
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(pw_bytes, hashed_bytes)


# ============================================================
# JWT TOKEN HANDLING
# ============================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": now})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency to retrieve the currently authenticated user."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    user_id: str = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ============================================================
# IOT DEVICE IDENTITY & API KEY
# ============================================================

def generate_device_api_key() -> str:
    """Generate a high-entropy secret API key for an IoT edge node."""
    return f"esp_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    """Compute SHA-256 digest of device API key."""
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()


def verify_api_key(plain_key: str, stored_hash: str) -> bool:
    """Constant-time comparison of hashed API key."""
    computed = hash_api_key(plain_key)
    return hmac.compare_digest(computed, stored_hash)


def get_authenticated_device(
    x_device_id: Optional[str] = Header(None, alias="X-Device-Id"),
    x_device_key: Optional[str] = Header(None, alias="X-Device-Key"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> IoTDevice:
    """
    Authenticate an ESP32 IoT device using:
    1. Dedicated headers: X-Device-Id and X-Device-Key, OR
    2. Bearer token in Authorization header formatted as: Bearer <device_id>:<device_key>
    """
    device_id = x_device_id
    device_key = x_device_key

    if not device_id and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
            if ":" in token:
                device_id, device_key = token.split(":", 1)
            else:
                device_key = token

    if not device_id or not device_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing device credentials. Provide X-Device-Id and X-Device-Key headers.",
        )

    device = (
        db.query(IoTDevice)
        .filter(IoTDevice.id == device_id, IoTDevice.is_active == True)
        .first()
    )

    if not device or not verify_api_key(device_key, device.api_key_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid device credentials",
        )

    # Update last seen timestamp
    device.last_seen_at = datetime.now(timezone.utc)
    db.commit()

    return device
