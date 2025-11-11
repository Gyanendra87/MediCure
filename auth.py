# auth.py
from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from database import User, SessionLocal
from datetime import datetime, timedelta
import jwt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ======================
# JWT Config
# ======================
JWT_SECRET = os.getenv("JWT_SECRET", "mysupersecretkey")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXP_DELTA_MINUTES = int(os.getenv("JWT_EXP_DELTA_MINUTES", 60))

# ======================
# Password hashing
# ======================
pwd_context = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")

# ======================
# Router
# ======================
auth_router = APIRouter(prefix="/auth", tags=["Auth"])

# ======================
# Database dependency
# ======================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ======================
# Register endpoint
# ======================
@auth_router.post("/register")
def register(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    try:
        # Check if user exists
        if db.query(User).filter(User.username == username).first():
            return JSONResponse(status_code=400, content={"detail": "Username already exists"})

        # Hash password
        hashed_password = pwd_context.hash(password[:72])
        user = User(username=username, password_hash=hashed_password)
        db.add(user)
        db.commit()
        db.refresh(user)

        return JSONResponse(content={"message": f"User {username} registered successfully"})
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"detail": str(e)})

# ======================
# Login endpoint with JWT
# ======================
@auth_router.post("/login")
def login(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user or not pwd_context.verify(password[:72], user.password_hash):
            return JSONResponse(status_code=401, content={"detail": "Invalid username or password"})

        # Generate JWT token
        payload = {
            "sub": user.username,
            "exp": datetime.utcnow() + timedelta(minutes=JWT_EXP_DELTA_MINUTES)
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

        # Return token
        return JSONResponse(content={
            "message": f"Welcome {username}! You are now logged in.",
            "access_token": token,
            "token_type": "bearer"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
