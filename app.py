import os
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import jwt
import base64
import bcrypt
import hashlib

from database import SessionLocal, init_db, User, Report

from graph_builder import graph


from disease_prediction import health_router


from home_remedies import predict_home_remedy, get_top_predictions, get_all_diseases


from uploadreport import uploadreport_router

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env!")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

JWT_SECRET = os.getenv("JWT_SECRET", "mysupersecretkey")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXP_DELTA_MINUTES = int(os.getenv("JWT_EXP_DELTA_MINUTES", 60))

app = FastAPI(title="MediCure AI Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
report_context = ""

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    sha256_hash = hashlib.sha256(password.encode("utf-8")).digest()
    hashed = bcrypt.hashpw(sha256_hash, bcrypt.gensalt())
    return base64.b64encode(hashed).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    sha256_hash = hashlib.sha256(password.encode("utf-8")).digest()
    hashed_bytes = base64.b64decode(hashed.encode("utf-8"))
    return bcrypt.checkpw(sha256_hash, hashed_bytes)

from auth import auth_router
auth_router.hash_password = hash_password
auth_router.verify_password = verify_password

chat_router = APIRouter(prefix="/chat", tags=["Chat"])
remedy_router = APIRouter(prefix="/remedies", tags=["Home Remedies"])
health_router_local = APIRouter(tags=["Health"])


@chat_router.get("/ask")
def ask_question(query: str = Query(...)):
    global report_context

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    remedy_keywords = ["remedy", "home remedy", "cure for", "treatment for"]
    if any(kw in query.lower() for kw in remedy_keywords):
        disease = query.split("for")[-1].replace("?", "").strip() if "for" in query.lower() else query.strip()
        result = predict_home_remedy(disease)
        yogasan_str = ", ".join([y["name"] for y in result["Yogasan"]]) if result["Yogasan"] else "N/A"

        answer = f"""
🌿 **Home Remedy for {result['Disease']}**

**Item:** {result['Item']}
**Home Remedy:** {result['HomeRemedy']}
**Yogasan:** {yogasan_str}
**Confidence:** {result['Confidence']}

{f"🔗 More Info: {result.get('Link', '')}"}
"""
        return {"answer": answer}

    state = {"messages": [{"role": "user", "content": query}], "context": report_context}
    response = graph.invoke(state)
    answer = response["messages"][-1].get("content", "No response")
    return {"answer": answer}


@remedy_router.get("/")
def get_home_remedy(disease: str = Query(...)):
    result = predict_home_remedy(disease)
    return {
        "disease": result["Disease"],
        "home_remedies": [result["HomeRemedy"]],
        "yogasan": result["Yogasan"],
        "source": result["Source"],
        "item": result["Item"],
        "image": result.get("Image", ""),
        "link": result.get("Link", ""),
        "confidence": result.get("Confidence", "N/A"),
    }

@remedy_router.get("/top")
def get_top_remedies(disease: str = Query(...), limit: int = Query(3, ge=1, le=10)):
    return {"disease": disease, "remedies": get_top_predictions(disease, top_n=limit)}

@remedy_router.get("/list")
def list_diseases():
    return {"diseases": get_all_diseases()}

@health_router_local.get("/health")
def health_check():
    return {"status": "ok", "message": "MediCure AI Backend is running"}


app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(uploadreport_router)
app.include_router(remedy_router)
app.include_router(health_router_local)
app.include_router(health_router)


if __name__ == "__main__":
    import uvicorn
    print(" MediCure AI Backend running at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
