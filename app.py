import os
import tempfile
from fastapi import FastAPI, APIRouter, Query, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, init_db, User, Report
from graph_builder import graph, predict_remedy
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from passlib.context import CryptContext

# ==========================
# FastAPI Setup
# ==========================
app = FastAPI(title="LangGraph Medical Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Password hasher
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Global variable for uploaded report
report_context = ""

# Routers
chat_router = APIRouter(prefix="/chat", tags=["Chat"])
report_router = APIRouter(prefix="/report", tags=["Report"])
auth_router = APIRouter(prefix="/auth", tags=["Auth"])

# Dependency for DB Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================
# Auth Routes (Register + Login)
# ==========================
@auth_router.post("/register")
def register(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    """
    Register a new user in the database
    """
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = pwd_context.hash(password)
    new_user = User(username=username, password_hash=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": f"User {username} registered successfully"}

@auth_router.post("/login")
def login(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    """
    Login and check credentials
    """
    user = db.query(User).filter(User.username == username).first()
    if not user or not pwd_context.verify(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {"message": f"Welcome {username}! You are now logged in."}

# ==========================
# Chatbot Route (QA + Home Remedies)
# ==========================
@chat_router.get("/ask")
async def ask_question(query: str = Query(...)):
    """
    Ask a medical question (no authentication needed)
    """
    global report_context
    try:
        # ✅ Home Remedies detection
        if "remedy" in query.lower() or "home remedy" in query.lower():
            disease_name = query.split("for")[-1].strip() if "for" in query.lower() else query
            remedy = predict_remedy(disease_name)
            answer = f"🌿 Home Remedy for {disease_name}: {remedy}"
            return {"answer": answer}

        # Default medical QA
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "context": report_context
        }
        result = graph.invoke(initial_state)
        answer = result["messages"][-1]["content"]
        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}

# ==========================
# PDF Report Upload
# ==========================
@report_router.post("/upload")
async def upload_report(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload PDF medical report (no authentication needed)
    """
    global report_context
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        report_context = "\n".join([chunk.page_content for chunk in chunks])

        # Generate summary using LLM
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=0.3,
            streaming=False
        )

        summaries = []
        for i, chunk in enumerate(chunks):
            prompt = (
                f"Summarize this section of a medical report ({i+1}/{len(chunks)}). "
                f"Focus on diagnosis, test results, and treatment suggestions:\n\n{chunk.page_content}"
            )
            response = llm.invoke(prompt)
            summaries.append(response.content)

        combined_prompt = (
            "Combine these partial medical report summaries into one clear and concise final summary:\n\n"
            + "\n\n".join(summaries)
        )
        final_response = llm.invoke(combined_prompt)
        summary_text = final_response.content if hasattr(final_response, "content") else str(final_response)

        # Store report summary in DB
        report_record = Report(file_name=file.filename, pdf_text=summary_text)
        db.add(report_record)
        db.commit()

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return {"summary": summary_text}

    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return {"error": str(e)}

# ==========================
# Include Routers
# ==========================
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(report_router)

print("✅ FastAPI Chatbot (DB + No Auth after Login) Ready!")
