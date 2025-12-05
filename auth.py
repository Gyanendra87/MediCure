from fastapi import APIRouter, Query, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import bcrypt
import hashlib
import base64
from database import User, SessionLocal
from datetime import datetime, timedelta
import jwt
import os
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "mysupersecretkey")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXP_DELTA_MINUTES = int(os.getenv("JWT_EXP_DELTA_MINUTES", 60))

auth_router = APIRouter(prefix="/auth", tags=["Auth"])

# DB Connection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Password hashing with SHA256 + bcrypt + Base64 storage
def hash_password(password: str) -> str:
    sha256_hash = hashlib.sha256(password.encode("utf-8")).digest()
    hashed = bcrypt.hashpw(sha256_hash, bcrypt.gensalt())
    return base64.b64encode(hashed).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    sha256_hash = hashlib.sha256(password.encode("utf-8")).digest()
    hashed_bytes = base64.b64decode(hashed.encode("utf-8"))
    return bcrypt.checkpw(sha256_hash, hashed_bytes)


# REGISTER
@auth_router.post("/register")
def register(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = hash_password(password)
    user = User(username=username, password_hash=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"message": f"User {username} registered successfully"}


# LOGIN
@auth_router.post("/login")
def login(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()

    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    payload = {
        "sub": user.username,
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXP_DELTA_MINUTES),
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return {"message": f"Welcome {username}!", "access_token": token, "token_type": "bearer"}
