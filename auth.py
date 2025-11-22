from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from database import User, SessionLocal
from datetime import datetime, timedelta
import jwt
import os
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "mysupersecretkey")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXP_DELTA_MINUTES = int(os.getenv("JWT_EXP_DELTA_MINUTES", 60))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

auth_router = APIRouter(prefix="/auth", tags=["Auth"])


# DB Connection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# REGISTER
@auth_router.post("/register")
def register(
    username: str = Query(...),
    password: str = Query(...),
    db: Session = Depends(get_db)
):
    if db.query(User).filter(User.username == username).first():
        return JSONResponse(status_code=400, content={"detail": "Username already exists"})

    hashed_password = pwd_context.hash(password)
    new_user = User(username=username, password_hash=hashed_password)

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": f"User {username} registered successfully"}


# LOGIN
@auth_router.post("/login")
def login(
    username: str = Query(...),
    password: str = Query(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()

    if not user:
        return JSONResponse(status_code=401, content={"detail": "Invalid username or password"})

    if not pwd_context.verify(password, user.password_hash):
        return JSONResponse(status_code=401, content={"detail": "Invalid username or password"})

    payload = {
        "sub": user.username,
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXP_DELTA_MINUTES)
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return {
        "message": "Login successful",
        "access_token": token,
        "token_type": "bearer"
    }
