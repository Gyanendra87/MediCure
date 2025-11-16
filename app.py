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
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# Routers
from health import health_router
# doctor router removed


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
# FASTAPI SETUP
# ==========================
app = FastAPI(title="MediCure AI Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
pwd_context = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")
report_context = ""


# ==========================
# DB Helper
# ==========================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Routers
auth_router = APIRouter(prefix="/auth", tags=["Auth"])
chat_router = APIRouter(prefix="/chat", tags=["Chat"])
report_router = APIRouter(prefix="/report", tags=["Report"])


# ==========================
# AUTH ROUTES
# ==========================
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

    payload = {"sub": user.username, "exp": datetime.utcnow() + timedelta(minutes=JWT_EXP_DELTA_MINUTES)}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return {"message": f"Welcome {username}!", "access_token": token, "token_type": "bearer"}


# ==========================
# CHAT ROUTE
# ==========================
@chat_router.get("/ask")
async def ask_question(query: str = Query(...)):
    global report_context

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Home remedy detection
    if any(k in query.lower() for k in ["remedy", "home remedy", "cure for", "treatment for"]):
        disease_name = query.split("for")[-1].strip() if "for" in query.lower() else query
        disease_name = disease_name.replace('"', '').replace(',', '').replace('?', '').strip()
        remedy = predict_remedy(disease_name)
        return {"answer": f"🌿 Home Remedy for {disease_name.title()}:\n\n{remedy}"}

    # Default chatbot
    state = {"messages": [{"role": "user", "content": query}], "context": report_context}
    response = graph.invoke(state)
    answer = response["messages"][-1].get("content", "No response")

    return {"answer": answer}


# ==========================
# REPORT UPLOAD
# ==========================
@report_router.post("/upload")
async def upload_report(file: UploadFile = File(...), db: Session = Depends(get_db)):
    global report_context
    tmp_path = None

    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        report_context = "\n".join([c.page_content for c in chunks])

        # Summarize
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        summaries = []

        for i, chunk in enumerate(chunks[:10]):
            prompt = f"Summarize section ({i+1}):\n{chunk.page_content[:2000]}"
            out = llm.invoke(prompt)
            summaries.append(out.content if hasattr(out, "content") else str(out))

        summary_text = " ".join(summaries)

        # Save summary in DB
        record = Report(file_name=file.filename, pdf_text=summary_text)
        db.add(record)
        db.commit()

        return {"summary": summary_text}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==========================
# ROUTER SETUP
# ==========================
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(report_router)
app.include_router(health_router)
# doctor router removed


# ==========================
# FRONTEND ROUTES
# ==========================
@app.get("/")
def serve_login():
    return FileResponse("login.html")


@app.get("/home")
def serve_home():
    return FileResponse("index.html")


# ==========================
# RUN (HUGGING FACE)
# ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
