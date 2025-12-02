import os
import tempfile
import traceback
import time
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
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
# HELPER FUNCTIONS
# ============================
def generate_fallback_summary(text: str, filename: str) -> str:
    """Generate a basic summary when AI is unavailable due to rate limits"""
    
    # Extract key information using simple text processing
    lines = text.split('\n')
    text_lower = text.lower()
    
    # Try to find key sections
    findings = []
    test_results = []
    diagnoses = []
    
    # Look for common medical keywords
    medical_keywords = ['diagnosis', 'findings', 'result', 'test', 'level', 'count', 'abnormal', 'normal']
    test_keywords = ['hemoglobin', 'wbc', 'rbc', 'glucose', 'cholesterol', 'pressure', 'rate', 'mg/dl', 'mmol/l']
    
    for line in lines[:50]:  # Check first 50 lines
        line_clean = line.strip()
        if not line_clean or len(line_clean) < 10:
            continue
            
        line_lower = line_clean.lower()
        
        # Capture lines with medical keywords
        if any(keyword in line_lower for keyword in medical_keywords):
            findings.append(line_clean[:100])  # Limit length
        
        # Capture lines with test keywords
        if any(keyword in line_lower for keyword in test_keywords):
            test_results.append(line_clean[:100])
    
    # Build fallback summary
    summary = f"""⚠️ BASIC SUMMARY (AI Rate Limited)

📄 **File:** {filename}
📊 **Document Length:** {len(text)} characters

## ⚡ Quick Extract
This is a basic text extraction. For full AI analysis, please try again in 1 minute.

"""
    
    if findings:
        summary += "## 🔍 Key Sections Found:\n"
        for finding in findings[:5]:
            summary += f"• {finding}\n"
        summary += "\n"
    
    if test_results:
        summary += "## 🧪 Test-Related Content:\n"
        for result in test_results[:5]:
            summary += f"• {result}\n"
        summary += "\n"
    
    # Add preview of full text
    summary += f"## 📋 Document Preview:\n{text[:500]}...\n\n"
    summary += "---\n"
    summary += "💡 **Note:** This is a basic text extraction due to API rate limits.\n"
    summary += "For AI-powered analysis with medical insights, please wait 60 seconds and upload again.\n"
    summary += "The document has been saved to the database (ID will be shown above)."
    
    return summary


# ============================
# ROUTER
# ============================
uploadreport_router = APIRouter(prefix="/report", tags=["Report"])


@uploadreport_router.post("/upload")
async def upload_medical_report(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process medical PDF reports - generates AI summary"""
    tmp_path = None

    try:
        # ============================================
        # STEP 1: VALIDATE FILE
        # ============================================
        print("=" * 50)
        print("🚀 STARTING PDF PROCESSING")
        print(f"📄 File: {file.filename}")
        print(f"📄 Content-Type: {file.content_type}")
        
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        # ============================================
        # STEP 2: SAVE FILE TEMPORARILY
        # ============================================
        print("💾 Saving file to temp location...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            print(f"📦 File size: {len(content)} bytes")
            
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="File is empty")
            
            tmp.write(content)
            tmp_path = tmp.name

        print(f"✅ Saved to: {tmp_path}")

        # ============================================
        # STEP 3: LOAD PDF
        # ============================================
        print("📖 Loading PDF with PyPDFLoader...")
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            raise HTTPException(status_code=400, detail="PDF is empty or unreadable")

        print(f"✅ Loaded {len(documents)} pages")
        
        # Debug: show first 200 chars of first page
        if documents:
            preview = documents[0].page_content[:200]
            print(f"📄 First page preview: {preview}...")

        # ============================================
        # STEP 4: SPLIT TEXT
        # ============================================
        print("✂️  Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise HTTPException(status_code=400, detail="No text found in PDF")

        print(f"✅ Created {len(chunks)} chunks")

        # ============================================
        # STEP 5: PREPARE TEXT FOR AI
        # ============================================
        # Take first 10 chunks (about 20,000 chars max)
        full_text = "\n".join(chunk.page_content for chunk in chunks[:10])
        
        # Limit to 8000 chars to stay within API limits
        full_text = full_text[:8000]
        
        print(f"📝 Text prepared: {len(full_text)} characters")
        print(f"📝 Preview: {full_text[:300]}...")

        # ============================================
        # STEP 6: CALL GEMINI AI (with fallback)
        # ============================================
        print("🤖 Initializing Gemini AI...")
        
        # Use stable model with better rate limits
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Stable model with generous free tier
            temperature=0.3,
            max_output_tokens=2048
        )

        prompt = f"""You are a medical expert AI assistant analyzing a medical report.

**Task:** Provide a clear, structured summary of the medical report below.

**Format your response as:**

## Key Findings
• [Important medical findings]

## Test Results
• [Laboratory values and results]

## Diagnosis
• [Any diagnoses mentioned]

## Critical Notes
• [Abnormal values or urgent findings]

**Medical Report Text:**
{full_text}

**Provide the summary now:**"""

        print("🤖 Sending request to Gemini API...")
        
        # Retry logic for rate limits with fallback
        max_retries = 2  # Reduced to 2 to fail faster
        retry_delay = 3
        final_summary = None
        
        for attempt in range(max_retries):
            try:
                result = llm.invoke(prompt)
                print("✅ Received response from Gemini!")
                
                # Extract summary
                if hasattr(result, 'content'):
                    if isinstance(result.content, str):
                        final_summary = result.content
                    elif isinstance(result.content, list):
                        final_summary = "".join([
                            item.get("text", "") if isinstance(item, dict) else str(item)
                            for item in result.content
                        ])
                    else:
                        final_summary = str(result.content)
                else:
                    final_summary = str(result)
                
                final_summary = final_summary.strip()
                break  # Success! Exit retry loop
                
            except Exception as api_error:
                error_msg = str(api_error)
                
                # Check if it's a rate limit error
                if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"⏳ Rate limit hit. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # All retries failed - use fallback summary
                        print("⚠️ All retries exhausted. Generating fallback summary...")
                        final_summary = generate_fallback_summary(full_text, file.filename)
                        break
                else:
                    # Other API error
                    print(f"❌ API Error: {error_msg}")
                    raise api_error
        
        # If somehow we still don't have a summary
        if not final_summary:
            final_summary = generate_fallback_summary(full_text, file.filename)
        
        print(f"✅ Summary generated: {len(final_summary)} characters")
        print(f"📋 Summary preview: {final_summary[:200]}...")
        
        if not final_summary or len(final_summary) < 20:
            # Fallback if summary is too short
            final_summary = generate_fallback_summary(full_text, file.filename)

        # ============================================
        # STEP 7: SAVE TO DATABASE
        # ============================================
        print("💾 Saving to database...")
        
        report_entry = Report(
            file_name=file.filename,
            pdf_text=final_summary
        )
        db.add(report_entry)
        db.commit()
        db.refresh(report_entry)

        print(f"✅ Saved to DB with ID: {report_entry.id}")
        print("=" * 50)

        # ============================================
        # STEP 8: RETURN RESPONSE
        # ============================================
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "PDF processed and summarized successfully",
                "file_name": file.filename,
                "report_id": report_entry.id,
                "summary": final_summary,
                "pages_processed": len(documents),
                "chunks_created": len(chunks)
            }
        )

    except HTTPException as he:
        print(f"❌ HTTP Exception: {he.detail}")
        raise he
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        print(f"❌ CRITICAL ERROR: {error_msg}")
        print(f"❌ Traceback:\n{error_trace}")
        
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Processing failed",
                "message": error_msg,
                "type": type(e).__name__
            }
        )

    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"🗑️  Cleaned up temp file: {tmp_path}")
            except Exception as e:
                print(f"⚠️  Could not remove temp file: {e}")


@uploadreport_router.get("/list")
async def list_reports(db: Session = Depends(get_db)):
    """Get all uploaded reports"""
    try:
        reports = db.query(Report).all()
        return {
            "success": True,
            "count": len(reports),
            "reports": [
                {
                    "id": r.id,
                    "file_name": r.file_name,
                    "uploaded_at": r.uploaded_at.isoformat() if hasattr(r, 'uploaded_at') else None
                }
                for r in reports
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@uploadreport_router.get("/{report_id}")
async def get_report(report_id: int, db: Session = Depends(get_db)):
    """Get a specific report summary by ID"""
    try:
        report = db.query(Report).filter(Report.id == report_id).first()
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {
            "success": True,
            "report": {
                "id": report.id,
                "file_name": report.file_name,
                "summary": report.pdf_text,
                "uploaded_at": report.uploaded_at.isoformat() if hasattr(report, 'uploaded_at') else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))