import os
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, Query, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt

# Database and models
from database import SessionLocal, init_db, User, Report

# Chatbot imports
from graph_builder import graph, predict_remedy

# Routers
from health import health_router
from uploadreport import uploadreport_router

# ==========================
# ENV SETUP
# ==========================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env!")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# JWT config
JWT_SECRET = os.getenv("JWT_SECRET", "mysupersecretkey")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXP_DELTA_MINUTES = int(os.getenv("JWT_EXP_DELTA_MINUTES", 60))

# ==========================
# FASTAPI SETUP WITH CORS
# ==========================
app = FastAPI(title="MediCure AI Chatbot")

# Add CORS middleware FIRST - this is critical
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# DATABASE SETUP
# ==========================
init_db()
pwd_context = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")
report_context = ""  # stores uploaded PDF summaries for chatbot context

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================
# AUTH ROUTER
# ==========================
auth_router = APIRouter(prefix="/auth", tags=["Auth"])

@auth_router.post("/register")
def register(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = pwd_context.hash(password[:72])
    user = User(username=username, password_hash=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"message": f"User {username} registered successfully"}

@auth_router.post("/login")
def login(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()

    if not user or not pwd_context.verify(password[:72], user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    payload = {
        "sub": user.username,
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXP_DELTA_MINUTES)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return {"message": f"Welcome {username}!", "access_token": token, "token_type": "bearer"}

# ==========================
# CHAT ROUTER
# ==========================
chat_router = APIRouter(prefix="/chat", tags=["Chat"])

@chat_router.get("/ask")
async def ask_question(query: str = Query(...)):
    global report_context
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Remedy detection
    if any(k in query.lower() for k in ["remedy", "home remedy", "cure for", "treatment for"]):
        disease_name = query.split("for")[-1].strip() if "for" in query.lower() else query
        disease_name = disease_name.replace('"', '').replace(',', '').replace('?', '').strip()
        remedy = predict_remedy(disease_name)
        return {"answer": f"🌿 Home Remedy for {disease_name.title()}:\n\n{remedy}"}

    # Chat with medical context
    state = {"messages": [{"role": "user", "content": query}], "context": report_context}
    response = graph.invoke(state)

    answer = response["messages"][-1].get("content", "No response")
    return {"answer": answer}

# ==========================
# HEALTH CHECK ROUTER
# ==========================
health_router_local = APIRouter(tags=["Health"])

@health_router_local.get("/health")
async def health_check():
    return {"status": "ok", "message": "MediCure AI Backend is running"}

# ==========================
# INCLUDE ALL ROUTERS
# ==========================
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(uploadreport_router)  # PDF upload router
app.include_router(health_router_local)
app.include_router(health_router)  # If you have external health router

# ==========================
# FRONTEND ROUTES
# ==========================
@app.get("/")
def serve_login():
    return FileResponse("frontend/login.html")

@app.get("/home")
def serve_home():
    return FileResponse("frontend/index.html")

# ==========================
# RUN SERVER
# ==========================
if __name__ == "__main__":
    import uvicorn
    print("🚀 MediCure AI Backend starting on http://0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=True)