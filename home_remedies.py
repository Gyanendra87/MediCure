import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote_plus
import requests
import warnings

warnings.filterwarnings('ignore')

# ==========================
# Google Custom Search API setup
# ==========================
GOOGLE_API_KEY = "AIzaSyAcs84HLFgqaFahv7gqeADpKPBvNySpEwo"
GOOGLE_CX = "5322ad2fa4e484776"

def fetch_google_results(disease_name, num_results=5):
    query = f"{disease_name} home remedy"
    url = f"https://www.googleapis.com/customsearch/v1?q={quote_plus(query)}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&num={num_results}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "link": item.get("link")
            })
        return results
    except Exception as e:
        return [{"title": "Error fetching Google results", "snippet": str(e), "link": ""}]

def extract_remedy_from_google(google_results):
    """Extract home remedy text from Google search results"""
    if not google_results:
        return "No remedies found online. Please consult a healthcare professional."
    
    # Combine all snippets into a remedy description
    remedies = []
    for i, result in enumerate(google_results[:3], 1):  # Use top 3 results
        snippet = result.get('snippet', '').strip()
        if snippet and snippet != '':
            remedies.append(f"{i}. {snippet}")
    
    if remedies:
        return "\n\n".join(remedies)
    else:
        return "No specific remedies found. Please consult a healthcare professional."

# ==========================
# Load remedies dataset
# ==========================
try:
    df = pd.read_csv("remedies.csv", encoding='utf-8', quotechar='"', skipinitialspace=True)
except Exception as e:
    raise FileNotFoundError("Error loading remedies.csv: " + str(e))

required_cols = ["Name of Item", "Disease", "Home Remedy", "Yogasan"]
for col in required_cols:
    if col not in df.columns:
        df[col] = ""

df["Disease"] = df["Disease"].astype(str).str.lower().str.strip()
df["Name of Item"] = df["Name of Item"].astype(str).str.strip()
df["Home Remedy"] = df["Home Remedy"].astype(str).str.strip()
df["Yogasan"] = df["Yogasan"].astype(str).str.strip()

if "Image" not in df.columns:
    df["Image"] = "https://via.placeholder.com/150"
if "Link" not in df.columns:
    df["Link"] = ""

df = df[(df["Disease"] != "") & (df["Home Remedy"] != "")].reset_index(drop=True)

# ==========================
# TF-IDF vectorizer
# ==========================
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True,
    stop_words='english'
)
disease_vectors = vectorizer.fit_transform(df["Disease"])

# ==========================
# Predict single remedy
# ==========================
def predict_home_remedy(disease_name):
    disease_name_clean = disease_name.lower().strip()
    
    # Exact match
    exact = df[df["Disease"] == disease_name_clean]
    if not exact.empty:
        match = exact.iloc[0]
        return {
            "Item": match["Name of Item"],
            "Disease": match["Disease"].title(),
            "HomeRemedy": match["Home Remedy"],
            "Yogasan": match["Yogasan"],
            "Image": match["Image"],
            "Link": match["Link"],
            "Source": "Database",
            "Confidence": "100%"
        }
    
    # Cosine similarity
    query_vector = vectorizer.transform([disease_name_clean])
    similarities = cosine_similarity(query_vector, disease_vectors)[0]
    best_idx = np.argmax(similarities)
    confidence = similarities[best_idx]
    
    if confidence >= 0.3:
        match = df.iloc[best_idx]
        return {
            "Item": match["Name of Item"],
            "Disease": match["Disease"].title(),
            "HomeRemedy": match["Home Remedy"],
            "Yogasan": match["Yogasan"],
            "Image": match["Image"],
            "Link": match["Link"],
            "Source": f"Database (Cosine {confidence:.2%})",
            "Confidence": f"{confidence:.2%}"
        }
    
    # Fallback to Google Custom Search
    google_results = fetch_google_results(disease_name)
    extracted_remedy = extract_remedy_from_google(google_results)
    
    return {
        "Item": "Web Search",
        "Disease": disease_name.title(),
        "HomeRemedy": extracted_remedy,
        "Yogasan": "Not available",
        "Image": "https://via.placeholder.com/150",
        "Link": google_results[0]["link"] if google_results else "",
        "Source": "Google Search Results",
        "GoogleResults": google_results,
        "Confidence": "N/A (External Source)"
    }

# ==========================
# Get top N remedies from CSV
# ==========================
def get_top_predictions(disease_name: str, top_n: int = 3):
    disease_name_clean = disease_name.lower().strip()
    query_vector = vectorizer.transform([disease_name_clean])
    similarities = cosine_similarity(query_vector, disease_vectors)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        match = df.iloc[idx]
        results.append({
            "Item": str(match["Name of Item"]),
            "Disease": str(match["Disease"]).title(),
            "HomeRemedy": str(match["Home Remedy"]),
            "Yogasan": str(match["Yogasan"]),
            "Image": str(match["Image"]),
            "Confidence": f"{similarities[idx]:.2%}"
        })
    return results

# ==========================
# Get all unique diseases
# ==========================
def get_all_diseases():
    return sorted(df["Disease"].unique().tolist())

# ==========================
# Test block
# ==========================
if __name__ == "__main__":
    test_diseases = ["cold", "fever", "piles", "unknown disease"]
    
    for disease in test_diseases:
        result = predict_home_remedy(disease)
        print(f"\n🔍 Disease: {disease}")
        print(f"Source: {result['Source']}")
        print(f"Item: {result['Item']}")
        print(f"Remedy: {result['HomeRemedy'][:200]}...")  # Show first 200 chars
        print(f"Yogasan: {result['Yogasan']}")
        print(f"Link: {result['Link']}")
        print(f"Confidence: {result['Confidence']}")
        
        if "GoogleResults" in result:
            print("\nGoogle Search Results:")
            for i, g in enumerate(result["GoogleResults"], 1):
                print(f"{i}. {g['title']}")
                print(f"   Link: {g['link']}")