import os
import tempfile
from dotenv import load_dotenv

# === CRITICAL: Load environment variables FIRST ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")

# Set environment variable BEFORE importing langchain
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# NOW import other modules
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
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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
    """Register a new user in the database"""
    try:
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")

        hashed_password = pwd_context.hash(password)
        new_user = User(username=username, password_hash=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"message": f"User {username} registered successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@auth_router.post("/login")
def login(username: str = Query(...), password: str = Query(...), db: Session = Depends(get_db)):
    """Login and check credentials"""
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user or not pwd_context.verify(password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid username or password")

        return {"message": f"Welcome {username}! You are now logged in."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

# ==========================
# Chatbot Route (QA + Home Remedies)
# ==========================
@chat_router.get("/ask")
async def ask_question(query: str = Query(...)):
    """Ask a medical question (no authentication needed)"""
    global report_context
    try:
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # ✅ Home Remedies detection
        if any(keyword in query.lower() for keyword in ["remedy", "home remedy", "home remedies", "treatment for", "cure for"]):
            # Extract disease name
            disease_name = query
            if "for" in query.lower():
                disease_name = query.split("for")[-1].strip()
            
            disease_name = disease_name.replace('"', '').replace(',', '').replace('?', '').strip()
            remedy = predict_remedy(disease_name)
            answer = f"🌿 Home Remedy for {disease_name.title()}:\n\n{remedy}"
            return {"answer": answer}

        # Default medical QA
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "context": report_context
        }
        
        result = graph.invoke(initial_state)
        
        # Extract answer from result
        if result and "messages" in result and len(result["messages"]) > 0:
            answer = result["messages"][-1].get("content", "No response generated")
        else:
            answer = "Sorry, I couldn't generate a response."
        
        return {"answer": answer}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return {"error": f"An error occurred: {str(e)}"}

# ==========================
# PDF Report Upload
# ==========================
@report_router.post("/upload")
async def upload_report(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload PDF medical report (no authentication needed)"""
    global report_context
    tmp_path = None
    
    try:
        # Validate file
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Load and process PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        report_context = "\n".join([chunk.page_content for chunk in chunks])

        # Generate summary using LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3
        )

        summaries = []
        max_chunks = min(len(chunks), 10)  # Limit to 10 chunks to avoid token issues
        
        for i, chunk in enumerate(chunks[:max_chunks]):
            prompt = (
                f"Summarize this section ({i+1}/{max_chunks}) of a medical report. "
                f"Focus on key findings, diagnosis, test results, and recommendations:\n\n"
                f"{chunk.page_content[:2000]}"  # Limit chunk size
            )
            response = llm.invoke(prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            summaries.append(summary)

        # Combine summaries
        if len(summaries) > 1:
            combined_prompt = (
                "Combine these medical report summaries into one clear, comprehensive summary. "
                "Include key diagnoses, test results, and treatment recommendations:\n\n"
                + "\n\n---\n\n".join(summaries)
            )
            final_response = llm.invoke(combined_prompt)
            summary_text = final_response.content if hasattr(final_response, 'content') else str(final_response)
        else:
            summary_text = summaries[0] if summaries else "Could not generate summary"

        # Store report summary in DB
        report_record = Report(file_name=file.filename, pdf_text=summary_text)
        db.add(report_record)
        db.commit()

        return {"summary": summary_text}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        db.rollback()
        return {"error": f"Failed to process report: {str(e)}"}
    finally:
        # Cleanup temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

# ==========================
# Include Routers
# ==========================
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(report_router)

# ==========================
# Health Check
# ==========================
@app.get("/")
def root():
    return {
        "message": "✅ MediCure API is running!",
        "endpoints": {
            "chat": "/chat/ask?query=your_question",
            "upload": "/report/upload",
            "register": "/auth/register",
            "login": "/auth/login"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("✅ FastAPI Chatbot (DB + No Auth after Login) Ready!")
    uvicorn.run(app, host="0.0.0.0", port=8000)