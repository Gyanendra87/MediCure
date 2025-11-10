import os
import tempfile
from fastapi import FastAPI, APIRouter, Query, UploadFile, File, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from graph_builder import graph, predict_remedy  # ✅ Now imports predict_remedy
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from auth import create_jwt_token, verify_jwt_token

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

# Global variable for last uploaded report
report_context = ""

# Routers
chat_router = APIRouter(prefix="/chat", tags=["Chat"])
report_router = APIRouter(prefix="/report", tags=["Report"])
auth_router = APIRouter(prefix="/auth", tags=["Auth"])

# ==========================
# Authentication Dependency
# ==========================
def get_current_user(authorization: str = Header(...)):
    """
    Verify JWT token in 'Authorization' header
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header format")
    token = authorization.split(" ")[1]
    try:
        payload = verify_jwt_token(token)
        return payload["user_id"]
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

# ==========================
# Hardcoded Users
# ==========================
USERS = {
    "user12": "mypassword123",
    "user1": "test123"
}

# ==========================
# Auth Routes
# ==========================
@auth_router.post("/login")
def login(user_id: str = Query(...), password: str = Query(...)):
    """
    Generate JWT token if user_id and password match.
    """
    if user_id not in USERS or USERS[user_id] != password:
        raise HTTPException(status_code=401, detail="Invalid user ID or password")
    token = create_jwt_token(user_id)
    return {"token": token}

# ==========================
# Chatbot Route (QA + Home Remedies)
# ==========================
@chat_router.get("/ask")
async def ask_question(query: str = Query(...), user_id: str = Depends(get_current_user)):
    """
    Ask a medical question. JWT required.
    """
    global report_context
    try:
        # ✅ Home Remedies detection
        if "remedy" in query.lower() or "home remedy" in query.lower():
            disease_name = query.split("for")[-1].strip() if "for" in query.lower() else query
            remedy = predict_remedy(disease_name)
            answer = f"🌿 Home Remedy for {disease_name}: {remedy}"
            return {"user": user_id, "answer": answer}

        # Default medical QA (uses medical_book.pdf)
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "context": report_context
        }
        result = graph.invoke(initial_state)
        answer = result["messages"][-1]["content"]
        return {"user": user_id, "answer": answer}

    except Exception as e:
        return {"error": str(e)}

# ==========================
# PDF Report Upload
# ==========================
@report_router.post("/upload")
async def upload_report(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    """
    Upload PDF medical report. JWT required.
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

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return {"user": user_id, "summary": summary_text}

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

print("✅ FastAPI Chatbot with JWT is ready!")
