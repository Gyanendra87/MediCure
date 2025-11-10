import os
from typing import TypedDict, List, Dict, Any
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
import google.generativeai as genai
from dotenv import load_dotenv

# === STEP 0: Load environment variables ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")
genai.configure(api_key=GOOGLE_API_KEY)

# === STEP 1: Load PDF for QA ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "medical_book.pdf")

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"❌ medical_book.pdf not found at {pdf_path}")

loader = PyPDFLoader(pdf_path)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# === STEP 2: Initialize Gemini LLM ===
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.3,
    streaming=False
)

# === STEP 3: Load Home Remedies CSV ===
remedies_file = os.path.join(BASE_DIR, "home_remedies.csv")
df = pd.read_csv(remedies_file)
df['Health Issue'] = df['Health Issue'].astype(str).str.strip().str.lower()
df['Home Remedy'] = df['Home Remedy'].astype(str).str.strip()

# === STEP 4: Function to get remedy from CSV ===
def predict_remedy(disease_name: str) -> str:
    key = disease_name.strip().lower()
    matched = df[df['Health Issue'] == key]
    if not matched.empty:
        return matched.iloc[0]['Home Remedy']
    else:
        return f"⚠️ No specific home remedy found for '{disease_name}'."

# === STEP 5: Define Chat State ===
class ChatState(TypedDict):
    messages: List[Dict[str, Any]]
    context: str

# === STEP 6: Graph Nodes ===
def retrieve_node(state: ChatState):
    query = state["messages"][-1]["content"]
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    state["context"] = context
    return state

def generate_node(state: ChatState):
    query = state["messages"][-1]["content"].strip()
    context = state.get("context", "")

    # 💊 Home Remedy detection
    if any(k in query.lower() for k in ["remedy", "home remedy", "home remedies", "treatment for", "cure for"]):
        disease_name = query.split("for")[-1].strip() if "for" in query.lower() else query
        disease_name = disease_name.replace('"', '').replace(',', '').title()
        remedy = predict_remedy(disease_name)
        state["messages"].append({
            "role": "assistant",
            "content": f"🌿 Home Remedy for {disease_name}:\n{remedy}"
        })
        return state

    # 🧠 Default QA using PDF context
    prompt = (
        f"Answer the medical query based on the following context from the medical book:\n\n"
        f"{context}\n\n"
        f"Question: {query}\nAnswer clearly and concisely:"
    )
    response = llm.invoke(prompt)
    state["messages"].append({"role": "assistant", "content": response.content})
    return state

# === STEP 7: Build and compile LangGraph ===
graph_builder = StateGraph(ChatState)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")

graph = graph_builder.compile()

print("✅ Chatbot ready with medical_book.pdf and home_remedies.csv")
