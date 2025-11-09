import os
from typing import TypedDict, List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader  # ✅ For PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
import google.generativeai as genai

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# === STEP 1: Load environment variables from .env ===
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")

genai.configure(api_key=GOOGLE_API_KEY)

# === STEP 2: Load medical_book.pdf for QA ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "medical_book.pdf")

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"❌ medical_book.pdf not found at {pdf_path}")

print("📘 Loading medical_book.pdf ...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# === STEP 3: Initialize Gemini LLM ===
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.3,
    streaming=False
)

# === STEP 4: Load Home Remedies Dataset & Train XGBoost ===
remedies_file = os.path.join(BASE_DIR, "home_remedies.csv")
df = pd.read_csv(remedies_file)

# Ensure proper column names exist
if "Health Issue" not in df.columns or "Home Remedy" not in df.columns:
    raise ValueError("❌ CSV file must contain 'Health Issue' and 'Home Remedy' columns.")

X = df['Health Issue'].astype(str).values
y = df['Home Remedy'].astype(str).values

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

xgb_model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False)
xgb_model.fit(X_tfidf, y_encoded)

# === STEP 5: Function to Predict Remedy ===
def predict_remedy(disease_name: str) -> str:
    try:
        x_input = tfidf.transform([disease_name])
        y_pred = xgb_model.predict(x_input)
        return le.inverse_transform(y_pred)[0]
    except Exception as e:
        return f"⚠️ Could not find a specific home remedy for {disease_name}. Error: {e}"

# === STEP 6: Define LangGraph State ===
class ChatState(TypedDict):
    messages: List[Dict[str, Any]]
    context: str

# === STEP 7: Define Graph Nodes ===
def retrieve_node(state: ChatState):
    query = state["messages"][-1]["content"]
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    state["context"] = context
    return state

def generate_node(state: ChatState):
    query = state["messages"][-1]["content"]
    context = state.get("context", "")

    # 💊 Home Remedy Detection
    if "remedy" in query.lower() or "home remedy" in query.lower():
        disease_name = query.split("for")[-1].strip() if "for" in query.lower() else query
        remedy = predict_remedy(disease_name)
        state["messages"].append({
            "role": "assistant",
            "content": f"🌿 Home Remedy for {disease_name}: {remedy}"
        })
        return state

    # 🧠 Default QA (context-aware using PDF)
    prompt = (
        f"Answer the medical query based on the following context from the medical book:\n\n"
        f"{context}\n\n"
        f"Question: {query}\nAnswer clearly and concisely:"
    )
    response = llm.invoke(prompt)
    state["messages"].append({"role": "assistant", "content": response.content})
    return state

# === STEP 8: Build and Compile LangGraph ===
graph_builder = StateGraph(ChatState)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")

graph = graph_builder.compile()
print("✅ Chatbot ready with medical_book.pdf and home_remedies.csv")
