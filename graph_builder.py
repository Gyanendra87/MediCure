import os
from typing import TypedDict, List, Dict, Any
import pandas as pd
from dotenv import load_dotenv

# === CRITICAL: Load environment variables FIRST ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# NOW import LangChain modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph, END

# === Initialize LLM (this fixes 'llm not defined') ===
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

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

# === STEP 3: Load Home Remedies CSV ===
remedies_file = os.path.join(BASE_DIR, "home_remedies.csv")
if not os.path.exists(remedies_file):
    raise FileNotFoundError(f"❌ home_remedies.csv not found at {remedies_file}")

df = pd.read_csv(remedies_file)
df['Disease'] = df['Disease'].astype(str).str.strip().str.lower()
df['Home Remedy'] = df['Home Remedy'].astype(str).str.strip()

# === STEP 4: Function to get remedy from CSV ===
def predict_remedy(disease_name: str) -> str:
    key = disease_name.strip().lower()
    matched = df[df['Disease'] == key]
    if not matched.empty:
        return matched.iloc[0]['Home Remedy']
    partial = df[df['Disease'].str.contains(key, na=False)]
    if not partial.empty:
        return partial.iloc[0]['Home Remedy']
    return f"⚠️ No specific home remedy found for '{disease_name}'. Please consult a healthcare professional."

# === STEP 5: Define Chat State ===
class ChatState(TypedDict):
    messages: List[Dict[str, Any]]
    context: str

# === STEP 6: Graph Nodes ===
def retrieve_node(state: ChatState) -> ChatState:
    try:
        query = state["messages"][-1]["content"]
        docs_found = retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs_found])
        state["context"] = context
    except Exception as e:
        print(f"⚠️ Retrieval error: {e}")
        state["context"] = ""
    return state

def generate_node(state: ChatState) -> ChatState:
    try:
        query = state["messages"][-1]["content"].strip()
        context = state.get("context", "")

        if any(k in query.lower() for k in ["remedy", "home remedy", "home remedies", "treatment for", "cure for"]):
            disease_name = query
            if "for" in query.lower():
                disease_name = query.lower().split("for")[-1].strip()
            disease_name = disease_name.replace('"','').replace(',','').replace('?','').strip()

            remedy = predict_remedy(disease_name)
            state["messages"].append({
                "role": "assistant",
                "content": f"🌿 Home Remedy for {disease_name.title()}:\n\n{remedy}"
            })
            return state

        prompt = (
            f"You are a helpful medical assistant. Answer the following medical query based on the context provided.\n\n"
            f"Context from medical book:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Provide a clear, concise, and helpful answer. If the context doesn't contain enough information, "
            f"say so and provide general medical knowledge if appropriate."
        )

        # ✅ Use the initialized LLM here
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        state["messages"].append({
            "role": "assistant",
            "content": answer
        })
    except Exception as e:
        print(f"⚠️ Generation error: {e}")
        state["messages"].append({
            "role": "assistant",
            "content": f"Sorry, I encountered an error: {str(e)}"
        })

    return state

# === STEP 7: Build and compile LangGraph ===
graph_builder = StateGraph(ChatState)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

print("✅ Chatbot ready with medical_book.pdf and home_remedies.csv (LLM fixed and loaded)")
