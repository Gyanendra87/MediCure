import os
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, Query, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import shutil

# Database + Models
from database import SessionLocal, init_db, User, Report
from graph_builder import graph
from disease_prediction import health_router  # Disease prediction routes

# Home Remedies (Cosine Similarity)
from home_remedies import predict_home_remedy, get_top_predictions, get_all_diseases

# ==========================
# Load environment variables
# ==========================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env!")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

JWT_SECRET = os.getenv("JWT_SECRET", "mysupersecretkey")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXP_DELTA_MINUTES = int(os.getenv("JWT_EXP_DELTA_MINUTES", 60))

# ==========================
# FastAPI + CORS
# ==========================
app = FastAPI(title="MediCure AI Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Database
# ==========================
init_db()
pwd_context = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")
report_context = ""  # Uploaded PDF text

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================
# Auth Router
# ==========================
auth_router = APIRouter(prefix="/auth", tags=["Auth"])

@auth_router.post("/register")
def register(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed = pwd_context.hash(password[:72])
    user = User(username=username, password_hash=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": f"User {username} registered successfully"}

@auth_router.post("/login")
def login(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not pwd_context.verify(password[:72], user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    payload = {"sub": user.username, "exp": datetime.utcnow() + timedelta(minutes=JWT_EXP_DELTA_MINUTES)}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"message": f"Welcome {username}!", "access_token": token, "token_type": "bearer"}

# ==========================
# Chat Router
# ==========================
chat_router = APIRouter(prefix="/chat", tags=["Chat"])

@chat_router.get("/ask")
def ask_question(query: str = Query(...)):
    global report_context
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    remedy_keywords = ["remedy", "home remedy", "cure for", "treatment for"]
    if any(kw in query.lower() for kw in remedy_keywords):
        disease = query.split("for")[-1].replace("?", "").strip() if "for" in query.lower() else query.strip()
        result = predict_home_remedy(disease)
        
        # Format yogasan nicely
        yogasan_str = ', '.join([y['name'] for y in result['Yogasan']]) if result['Yogasan'] else 'N/A'
        
        answer = f"""
🌿 **Home Remedy for {result['Disease']}**

**Item:** {result['Item']}
**Home Remedy:** {result['HomeRemedy']}
**Yogasan:** {yogasan_str}
**Confidence:** {result['Confidence']}

{f"🔗 More Info: {result['Link']}" if result.get('Link') else ""}
"""
        return {"answer": answer}

    # Default: Graph-based chatbot response
    state = {"messages": [{"role": "user", "content": query}], "context": report_context}
    response = graph.invoke(state)
    answer = response["messages"][-1].get("content", "No response")
    return {"answer": answer}

# ==========================
# PDF Upload Router
# ==========================
upload_router = APIRouter(prefix="/upload", tags=["Upload"])

@upload_router.post("/")
def upload_pdf(file: UploadFile = File(...)):
    global report_context
    file_location = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    report_context = f"PDF uploaded: {file.filename}"
    return {"message": f"Uploaded {file.filename}"}

# ==========================
# Home Remedies API
# ==========================
remedy_router = APIRouter(prefix="/remedies", tags=["Home Remedies"])

@remedy_router.get("/")
def get_home_remedy(disease: str = Query(...)):
    """
    Get home remedies for a specific disease using cosine similarity.
    
    Args:
        disease: Name of the disease
        
    Returns:
        JSON with disease, home_remedies, yogasan (with links), source, and confidence
    """
    if not disease.strip():
        raise HTTPException(status_code=400, detail="Disease name cannot be empty")
    
    try:
        result = predict_home_remedy(disease)
        
        # Log the prediction for debugging
        print(f"✓ Query: '{disease}' -> Matched: '{result['Disease']}' -> Item: '{result['Item']}' -> Confidence: {result['Confidence']}")
        
        # Return in the format expected by frontend
        return {
            "disease": result['Disease'],
            "home_remedies": [result['HomeRemedy']],
            "yogasan": result['Yogasan'],
            "source": result['Source'],
            "item": result['Item'],
            "image": result.get('Image', ''),
            "link": result.get('Link', ''),
            "confidence": result.get('Confidence', 'N/A')
        }
    except Exception as e:
        print(f"❌ Error predicting remedy for '{disease}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error predicting remedy: {str(e)}")

@remedy_router.get("/top")
def get_top_remedies(disease: str = Query(...), limit: int = Query(3, ge=1, le=10)):
    """
    Get top N remedies for a disease.
    
    Args:
        disease: Name of the disease
        limit: Number of top results (1-10)
        
    Returns:
        List of top remedies with confidence scores
    """
    if not disease.strip():
        raise HTTPException(status_code=400, detail="Disease name cannot be empty")
    
    try:
        results = get_top_predictions(disease, top_n=limit)
        return {"disease": disease, "remedies": results, "count": len(results)}
    except Exception as e:
        print(f"❌ Error getting top remedies for '{disease}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting top remedies: {str(e)}")

@remedy_router.get("/list")
def list_diseases():
    """
    Get list of all available diseases in the database.
    
    Returns:
        List of disease names
    """
    try:
        diseases = get_all_diseases()
        return {"diseases": diseases, "count": len(diseases)}
    except Exception as e:
        print(f"❌ Error listing diseases: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing diseases: {str(e)}")

# ==========================
# Health Check Router
# ==========================
health_router_local = APIRouter(tags=["Health"])

@health_router_local.get("/health")
def health_check():
    return {"status": "ok", "message": "MediCure AI Backend is running"}

# ==========================
# Include Routers
# ==========================
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(remedy_router)
app.include_router(health_router_local)
app.include_router(health_router)  # Disease prediction module

# ==========================
# Frontend Serving
# ==========================
@app.get("/")
def login_page():
    return FileResponse("frontend/login.html")

@app.get("/home")
def home_page():
    return FileResponse("frontend/index.html")

# ==========================
# Run server
# ==========================
if __name__ == "__main__":
    import uvicorn
    print("🚀 MediCure AI Backend running at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)