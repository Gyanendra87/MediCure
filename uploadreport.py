import os
import tempfile
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

from database import SessionLocal, Report


# ============================
# ENV SETUP
# ============================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env!")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================
# ROUTER
# ============================
uploadreport_router = APIRouter(prefix="/report", tags=["Report"])


@uploadreport_router.post("/upload")
async def upload_medical_report(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process medical PDF reports"""
    tmp_path = None

    try:
        # Validate PDF
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            raise HTTPException(status_code=400, detail="PDF is empty or unreadable")

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise HTTPException(status_code=400, detail="No text found in PDF")

        # Combine all chunks into one text → **1 API call only**
        full_text = "\n".join(chunk.page_content for chunk in chunks[:10])

        # Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2
        )

        prompt = f"""
        You are a medical expert.

        Summarize the following medical report into very clear bullet points.
        Focus on:
        • Important medical findings  
        • Test results  
        • Abnormal values  
        • Any diagnosis mentioned  
        • Anything critical  

        Report:
        {full_text}

        Respond ONLY with bullet points.
        """

        result = llm.invoke(prompt)

        # Extract summary text
        if isinstance(result.content, list):
            final_summary = "".join([p["text"] for p in result.content if "text" in p])
        else:
            final_summary = str(result.content)

        # Save to DB
        report_entry = Report(
            file_name=file.filename,
            pdf_text=final_summary
        )
        db.add(report_entry)
        db.commit()

        return {
            "message": "PDF processed successfully",
            "summary": final_summary,
            "file_name": file.filename
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
