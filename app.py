import os
import tempfile
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from graph_builder import graph
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================
# 🚀 FASTAPI SETUP
# ==========================
app = FastAPI(title="LangGraph Medical Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# GLOBAL VARIABLE TO STORE LAST UPLOADED REPORT CONTENT
# ==========================
report_context = ""

# ==========================
# 🩺 BASIC ROUTE
# ==========================
@app.get("/")
def root():
    return {"message": "✅ LangGraph Medical Chatbot is running!"}


# ==========================
# 💬 CHATBOT ROUTE (QA + Remedies)
# ==========================
@app.get("/ask")
async def ask_question(query: str = Query(...)):
    """
    Ask a medical question and get an answer from LangGraph + Gemini.
    If a report was uploaded, include its content as context.
    """
    try:
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "context": report_context  # Include uploaded report if any
        }
        result = graph.invoke(initial_state)
        answer = result["messages"][-1]["content"]
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


# ==========================
# 📄 PDF REPORT SUMMARIZER
# ==========================
@app.post("/upload-report/")
async def upload_report(file: UploadFile = File(...)):
    """
    Upload a PDF medical report and get a summarized analysis.
    The content is also saved for follow-up questions.
    """
    global report_context

    try:
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Split into manageable chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # Store report content for follow-up questions
        report_context = "\n".join([chunk.page_content for chunk in chunks])

        # Initialize LLM for summarization
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=0.3,
            streaming=False
        )

        # Summarize each chunk
        summaries = []
        for i, chunk in enumerate(chunks):
            prompt = (
                f"Summarize this section of a medical report ({i+1}/{len(chunks)}). "
                f"Focus on diagnosis, test results, and treatment suggestions:\n\n{chunk.page_content}"
            )
            response = llm.invoke(prompt)
            summaries.append(response.content)

        # Combine all summaries into one concise final summary
        combined_prompt = (
            "Combine these partial medical report summaries into one clear and concise final summary:\n\n"
            + "\n\n".join(summaries)
        )
        final_response = llm.invoke(combined_prompt)

        # Extract the final summary text safely
        summary_text = (
            final_response.content
            if hasattr(final_response, "content")
            else str(final_response)
        )

        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return JSONResponse({"summary": summary_text})

    except Exception as e:
        # Safe cleanup even if error occurs
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return JSONResponse({"error": str(e)})
