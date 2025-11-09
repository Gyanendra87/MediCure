import os
from typing import TypedDict, List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
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

# === STEP 1: Load environment variables ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")

genai.configure(api_key=GOOGLE_API_KEY)

# === STEP 2: Load medical.txt for general QA ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
medical_file = os.path.join(BASE_DIR, "data", "medical.txt")

loader = TextLoader(medical_file, encoding="utf-8")
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

# === STEP 4: Load Home Remedies Dataset ===
remedies_file = os.path.join(BASE_DIR, "home_remedies.csv")
df = pd.read_csv(remedies_file)

# Normalize text
df["Health Issue"] = df["Health Issue"].astype(str).str.lower().str.strip()
df["Home Remedy"] = df["Home Remedy"].astype(str)
df["Name of Item"] = df["Name of Item"].astype(str)

# === STEP 5: Train TF-IDF + XGBoost ===
X = df["Health Issue"].values
y = df["Home Remedy"].values

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

xgb_model = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", use_label_encoder=False)
xgb_model.fit(X_tfidf, y_encoded)

# === STEP 6: Predict Remedy ===
def predict_remedy(disease_name: str) -> str:
    disease_name = disease_name.lower().strip()

    # 🔹 Direct CSV match
    if disease_name in df["Health Issue"].values:
        row = df.loc[df["Health Issue"] == disease_name].iloc[0]
        return f"💊 Home Remedy for {disease_name.title()}: Use {row['Name of Item']} — {row['Home Remedy']}"

    # 🔹 Predict if not in CSV
    x_input = tfidf.transform([disease_name])
    y_pred = xgb_model.predict(x_input)
    predicted_remedy = le.inverse_transform(y_pred)[0]

    # 🔹 Fallback safe suggestion
    return f"🌿 For {disease_name.title()}, rest well, stay hydrated, and try mild natural treatments like turmeric paste or cold compress if swelling persists."

# === STEP 7: Define LangGraph State ===
class ChatState(TypedDict):
    messages: List[Dict[str, Any]]
    context: str

# === STEP 8: Define Nodes ===
def retrieve_node(state: ChatState):
    query = state["messages"][-1]["content"]
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    state["context"] = context
    return state


def generate_node(state: ChatState):
    query = state["messages"][-1]["content"].lower()
    context = state.get("context", "")

    # 🔍 If the query matches or mentions any health issue from CSV — treat as remedy request
    for issue in df["Health Issue"].values:
        if issue in query:
            remedy_text = predict_remedy(issue)
            state["messages"].append({"role": "assistant", "content": f"🌿 {remedy_text}"})
            return state

    # 🔍 If query explicitly asks for a remedy
    if any(word in query for word in ["remedy", "cure", "treatment", "home remedy"]):
        disease_name = query.split("for")[-1].strip() if "for" in query else query
        remedy_text = predict_remedy(disease_name)
        state["messages"].append({"role": "assistant", "content": f"🌿 {remedy_text}"})
        return state

    # 🧠 Otherwise use LLM + medical.txt
    prompt = (
        f"Answer the medical query based on the following context:\n\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    response = llm.invoke(prompt)
    state["messages"].append({"role": "assistant", "content": response.content})
    return state

# === STEP 9: Build Graph ===
graph_builder = StateGraph(ChatState)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")

graph = graph_builder.compile()
