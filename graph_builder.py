import os
from typing import TypedDict, List, Dict, Any
import pandas as pd
from dotenv import load_dotenv

# LangChain + LangGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph, END


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "medical_book.pdf")

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f" medical_book.pdf not found at {pdf_path}")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

remedies_csv = os.path.join(BASE_DIR, "remedies.csv")
if not os.path.exists(remedies_csv):
    raise FileNotFoundError(f" remedies.csv not found at {remedies_csv}")

df = pd.read_csv(remedies_csv)
df["Disease"] = df["Disease"].astype(str).str.strip().str.lower()
df["Home Remedy"] = df["Home Remedy"].astype(str).str.strip()

def predict_remedy(disease_name: str) -> str:
    """Fetch home remedy from remedies dataframe."""
    key = disease_name.strip().lower()

    matched = df[df["Disease"] == key]
    if not matched.empty:
        return matched.iloc[0]["Home Remedy"]

    partial = df[df["Disease"].str.contains(key, na=False)]
    if not partial.empty:
        return partial.iloc[0]["Home Remedy"]

    return f"No home remedy found for '{disease_name}'."


# Chat State

class ChatState(TypedDict):
    messages: List[Dict[str, Any]]
    context: str


# LangGraph Nodes

def retrieve_node(state: ChatState) -> ChatState:
    """Retrieve relevant docs from PDF using FAISS."""
    try:
        query = state["messages"][-1]["content"]
        docs_found = retriever.invoke(query)

        state["context"] = "\n".join([d.page_content for d in docs_found])

    except Exception as e:
        print(f"Retrieval Error: {e}")
        state["context"] = ""

    return state


def generate_node(state: ChatState) -> ChatState:
    """Generate response using LLM or remedies dataset."""
    try:
        query = state["messages"][-1]["content"].strip()
        context = state.get("context", "")

      
        if any(k in query.lower() for k in ["remedy", "home remedy", "treatment for", "cure for"]):
            disease_name = query.lower()
            if "for" in disease_name:
                disease_name = disease_name.split("for")[-1].strip()

            remedy = predict_remedy(disease_name)
            state["messages"].append({
                "role": "assistant",
                "content": f"Home Remedy for {disease_name.title()}:\n{remedy}"
            })
            return state

      
        prompt = (
            "You are a helpful medical assistant. Use the following context "
            "from a medical book to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "If context is insufficient, provide general medical guidance."
        )

        response = llm.invoke(prompt)
        answer = getattr(response, "content", str(response))

        state["messages"].append({
            "role": "assistant",
            "content": answer
        })

    except Exception as e:
        print(f"Generation Error: {e}")
        state["messages"].append({
            "role": "assistant",
            "content": f"⚠ Error: {str(e)}"
        })

    return state

graph_builder = StateGraph(ChatState)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

print("Chatbot ready with PDF + Remedies dataset loaded!")
