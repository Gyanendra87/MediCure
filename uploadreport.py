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

I
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    print(" google-generativeai library loaded")
except ImportError as e:
    GENAI_AVAILABLE = False
    print(f" google-generativeai not available: {e}")

from database import SessionLocal, Report


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY and GENAI_AVAILABLE:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print(" Google Generative AI configured successfully")
    except Exception as e:
        print(f" Failed to configure Google AI: {e}")
        GENAI_AVAILABLE = False
elif not GOOGLE_API_KEY:
    print(" GOOGLE_API_KEY not found in environment variables")
    GENAI_AVAILABLE = False


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_fallback_summary(text: str, filename: str) -> str:
    """Generate a basic summary when AI is unavailable"""
    lines = text.split('\n')
    findings = []
    test_results = []
    
    medical_keywords = ['diagnosis', 'findings', 'result', 'test', 'level', 'count', 'abnormal', 'normal', 'patient', 'doctor', 'hospital']
    test_keywords = ['hemoglobin', 'wbc', 'rbc', 'glucose', 'cholesterol', 'pressure', 'rate', 'mg/dl', 'mmol/l', 'bpm', 'temperature']

    for line in lines[:50]:
        line_clean = line.strip()
        if not line_clean or len(line_clean) < 10:
            continue
        line_lower = line_clean.lower()
        
        if any(keyword in line_lower for keyword in medical_keywords):
            if line_clean not in findings and len(findings) < 5:
                findings.append(line_clean[:120])
        
        if any(keyword in line_lower for keyword in test_keywords):
            if line_clean not in test_results and len(test_results) < 5:
                test_results.append(line_clean[:120])

    summary = f"""# Document Summary (Basic Extraction)

 **File:** {filename}  
 **Document Length:** {len(text):,} characters  
 **Note:** AI analysis unavailable - showing basic text extraction

"""

    if findings:
        summary += "##  Key Sections Detected:\n\n"
        for i, finding in enumerate(findings, 1):
            summary += f"{i}. {finding}\n"
        summary += "\n"
    else:
        summary += "##  Key Sections Detected:\n\nNo specific medical keywords found.\n\n"

    if test_results:
        summary += "##  Test-Related Content:\n\n"
        for i, result in enumerate(test_results, 1):
            summary += f"{i}. {result}\n"
        summary += "\n"
    else:
        summary += "##  Test-Related Content:\n\nNo test-related content detected.\n\n"

    # Show preview of document
    preview_text = text[:1000] if len(text) > 1000 else text
    summary += f"## 📋 Document Preview:\n\n```\n{preview_text}\n```\n\n"
    
    if len(text) > 1000:
        summary += f"*...and {len(text) - 1000:,} more characters*\n\n"
    
    summary += "---\n\n"
    summary += " **To enable AI-powered analysis:**\n"
    summary += "1. Get an API key from https://makersuite.google.com/app/apikey\n"
    summary += "2. Add it to your .env file as GOOGLE_API_KEY=your_key_here\n"
    summary += "3. Restart the application\n"
    
    return summary

def generate_ai_summary(text: str, filename: str) -> tuple:
    """
    Generate AI summary using Google Generative AI
    Returns: (summary_text, model_name) or (None, None) if failed
    """
    if not GENAI_AVAILABLE or not GOOGLE_API_KEY:
        print("AI summary not available - missing API key or library")
        return None, None
    
    # Models to try in order of preference
    model_options = [
        "gemini-1.5-flash",
        "gemini-1.5-pro", 
        "gemini-pro",
        "gemini-1.0-pro"
    ]
    
    max_retries = 2
    
    for model_name in model_options:
        print(f" Trying model: {model_name}")
        
        for attempt in range(max_retries):
            try:
                # Initialize model
                model = genai.GenerativeModel(model_name)
                
                # Create prompt
                prompt = f"""You are a medical AI assistant. Analyze this medical document and provide a clear, structured summary.

**IMPORTANT:** Format your response EXACTLY as shown below with these section headers:

## Key Findings
[List the main medical findings, diagnoses, or observations]

## Test Results  
[List any laboratory values, measurements, or test results]

## Recommendations
[List any medical advice, prescriptions, or follow-up instructions]

## Critical Notes
[List any abnormal values, urgent findings, or important warnings]

---

**Document to analyze:**

{text}

---

Provide your structured summary now:"""

                print(f"    Sending request to {model_name} (attempt {attempt + 1}/{max_retries})...")
          
                generation_config = {
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                    "candidate_count": 1,
                }
                
         
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                
                # Generate content
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
            
                summary_text = None
                
                if response and hasattr(response, 'text'):
                    try:
                        summary_text = response.text.strip()
                    except Exception as text_error:
                        print(f"   ⚠️ Could not access .text property: {text_error}")
                
                
                if not summary_text and hasattr(response, 'candidates'):
                    try:
                        if response.candidates and len(response.candidates) > 0:
                            candidate = response.candidates[0]
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                parts_text = ''.join([
                                    part.text for part in candidate.content.parts 
                                    if hasattr(part, 'text')
                                ])
                                if parts_text:
                                    summary_text = parts_text.strip()
                    except Exception as candidate_error:
                        print(f"    Could not extract from candidates: {candidate_error}")
                
               
                if summary_text and len(summary_text) > 100:
                    print(f"Successfully generated summary with {model_name}")
                    print(f"   Summary length: {len(summary_text):,} characters")
                    return summary_text, model_name
                else:
                    print(f" Response too short or empty: {len(summary_text) if summary_text else 0} chars")
                
            except Exception as api_error:
                err_msg = str(api_error).lower()
                print(f" Error with {model_name} (attempt {attempt + 1}/{max_retries}):")
                print(f"   Error: {str(api_error)[:250]}")
                
                # Handle specific error types
                if any(x in err_msg for x in ["404", "not found", "does not exist", "not supported"]):
                    print(f" Model {model_name} not available, trying next model...")
                    break  
                    
                elif any(x in err_msg for x in ["429", "rate", "quota", "resource_exhausted", "resource exhausted"]):
                    if attempt < max_retries - 1:
                        wait_time = 3 * (2 ** attempt)  
                        print(f" Rate limit hit. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                        print(f" Rate limit exceeded for {model_name}, trying next model...")
                        break  
                        
                elif any(x in err_msg for x in ["api key", "invalid", "authentication", "permission denied", "permission_denied"]):
                    print(f" API authentication failed!")
                    print(f"   Please check your API key at: https://makersuite.google.com/app/apikey")
                    return None, None  
                    
                elif any(x in err_msg for x in ["blocked", "safety"]):
                    print(f" Content blocked by safety filters, trying next model...")
                    break  
                else:
                  
                    if attempt < max_retries - 1:
                        print(f"⏳ Retrying in 2 seconds...")
                        time.sleep(2)
                        continue 
                        print(f" All retries exhausted for {model_name}, trying next model...")
                        break 
    
    print("All AI models failed or unavailable")
    return None, None


uploadreport_router = APIRouter(prefix="/report", tags=["Report"])

@uploadreport_router.post("/upload")
async def upload_medical_report(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process medical PDF reports - generates AI summary"""
    tmp_path = None

    try:
       
        print(" PDF PROCESSING STARTED")
      
        print(f"File: {file.filename}")

        # Validate file
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

      
        print(" Saving file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="File is empty")
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"   Temp path: {tmp_path}")
        print(f"   File size: {len(content):,} bytes")

        # Load PDF
        print(" Loading PDF pages...")
        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
        except Exception as load_error:
            print(f" PDF loading failed: {load_error}")
            raise HTTPException(status_code=400, detail=f"Failed to load PDF: {str(load_error)}")
        
        if not documents:
            raise HTTPException(status_code=400, detail="PDF appears to be empty or corrupted")
        
        print(f"Loaded {len(documents)} page(s)")

        
        print(" Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text content found in PDF")
        
        print(f" Created {len(chunks)} chunk(s)")
        
        full_text = "\n\n".join(chunk.page_content for chunk in chunks[:10])
        if len(full_text) > 10000:
            full_text = full_text[:10000]
            print(f"📝 Text truncated to 10,000 characters for analysis")
        
        print(f"📝 Prepared text: {len(full_text):,} characters")

        
        
        final_summary = None
        model_used = "fallback"
        
        
        if GENAI_AVAILABLE and GOOGLE_API_KEY:
            ai_summary, model_name = generate_ai_summary(full_text, file.filename)
            if ai_summary:
                final_summary = ai_summary
                model_used = model_name
                print(f"AI summary generated successfully")
            else:
                print("AI summary failed, using fallback...")
        else:
            print(" AI not available, using fallback summary...")
        
        if not final_summary:
            final_summary = generate_fallback_summary(full_text, file.filename)
            model_used = "fallback"
            print(f" Fallback summary generated")

    
        print("\nSaving to database...")
        try:
            report_entry = Report(
                file_name=file.filename, 
                pdf_text=final_summary
            )
            db.add(report_entry)
            db.commit()
            db.refresh(report_entry)
            print(f"Saved with ID: {report_entry.id}")
        except Exception as db_error:
            db.rollback()
            print(f" Database error: {db_error}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        
        print("=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70 + "\n")

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "PDF processed successfully",
                "file_name": file.filename,
                "report_id": report_entry.id,
                "summary": final_summary,
                "pages_processed": len(documents),
                "chunks_created": len(chunks),
                "ai_model_used": model_used,
                "text_length": len(full_text)
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        print(f"\nCRITICAL ERROR:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "type": type(e).__name__,
                "message": "Failed to process PDF. Please check the file and try again."
            }
        )

    finally:
   
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"🗑️ Cleaned up temp file: {tmp_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Could not remove temp file: {cleanup_error}")


# LIST REPORTS

@uploadreport_router.get("/list")
async def list_reports(db: Session = Depends(get_db)):
    """Get list of all uploaded reports"""
    try:
        reports = db.query(Report).order_by(Report.id.desc()).all()
        
        return {
            "success": True,
            "count": len(reports),
            "reports": [
                {
                    "id": r.id,
                    "file_name": r.file_name,
                    "uploaded_at": r.uploaded_at.isoformat() if hasattr(r, 'uploaded_at') and r.uploaded_at else None,
                    "summary_preview": (r.pdf_text[:200] + "...") if r.pdf_text and len(r.pdf_text) > 200 else r.pdf_text
                } 
                for r in reports
            ]
        }
    except Exception as e:
        print(f"Error listing reports: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")


# GET REPORT BY ID

@uploadreport_router.get("/{report_id}")
async def get_report(report_id: int, db: Session = Depends(get_db)):
    """Get a specific report by ID"""
    try:
        report = db.query(Report).filter(Report.id == report_id).first()
        
        if not report:
            raise HTTPException(
                status_code=404, 
                detail=f"Report with ID {report_id} not found"
            )
        
        return {
            "success": True,
            "report": {
                "id": report.id,
                "file_name": report.file_name,
                "summary": report.pdf_text,
                "uploaded_at": report.uploaded_at.isoformat() if hasattr(report, 'uploaded_at') and report.uploaded_at else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving report {report_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve report: {str(e)}")


@uploadreport_router.delete("/{report_id}")
async def delete_report(report_id: int, db: Session = Depends(get_db)):
    """Delete a report by ID"""
    try:
        report = db.query(Report).filter(Report.id == report_id).first()
        
        if not report:
            raise HTTPException(
                status_code=404, 
                detail=f"Report with ID {report_id} not found"
            )
        
        db.delete(report)
        db.commit()
        
        print(f"Deleted report {report_id}")
        
        return {
            "success": True,
            "message": f"Report {report_id} ({report.file_name}) deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting report {report_id}: {e}")
        db.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete report: {str(e)}")