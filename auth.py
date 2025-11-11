# auth.py
import os
from fastapi import FastAPI, APIRouter, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
from dotenv import load_dotenv
from database import SessionLocal, init_db, User

# ==========================
# Load environment variables
# ==========================
load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXP_DELTA_MINUTES = int(os.getenv("JWT_EXP_DELTA_MINUTES", 60))
MAX_PASSWORD_LENGTH = 72

# ==========================
# FastAPI setup
# ==========================
app = FastAPI(title="Auth API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Database setup
# ==========================
init_db()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================
# Password hasher
# ==========================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    truncated = password[:MAX_PASSWORD_LENGTH]
    return pwd_context.hash(truncated)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    truncated = plain_password[:MAX_PASSWORD_LENGTH]
    return pwd_context.verify(truncated, hashed_password)

# ==========================
# JWT helpers
# ==========================
def create_jwt_token(user_id: str):
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXP_DELTA_MINUTES)
    payload = {
        "user_id": user_id,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def verify_jwt_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ==========================
# Auth router
# ==========================
auth_router = APIRouter(prefix="/auth", tags=["Auth"])

@auth_router.post("/register")
def register(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    """Register a new user"""
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed = hash_password(password)
    new_user = User(username=username, password_hash=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": f"User '{username}' registered successfully"}

@auth_router.post("/login")
def login(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    """Login and return JWT token"""
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_jwt_token(user.username)
    return {"token": token, "message": f"Welcome {username}! You are now logged in."}

# ==========================
# Include router if running standalone
# ==========================
app.include_router(auth_router)

if __name__ == "__main__":
    import uvicorn
    print("✅ Auth API Ready!")
    uvicorn.run(app, host="0.0.0.0", port=8000)
