import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote_plus
import requests
import warnings
import re

warnings.filterwarnings('ignore')

# ==========================
# Google Custom Search API setup
# ==========================
GOOGLE_API_KEY = "AIzaSyAcs84HLFgqaFahv7gqeADpKPBvNySpEwo"
GOOGLE_CX = "5322ad2fa4e484776"

def is_english(text):
    """Check if text is primarily English (basic check)"""
    if not text:
        return False
    # Check for non-ASCII characters that indicate non-English text
    non_ascii = len([c for c in text if ord(c) > 127])
    # If more than 20% non-ASCII, likely not English
    return (non_ascii / len(text)) < 0.2 if len(text) > 0 else True

def clean_remedy_text(text):
    """Clean and format remedy text"""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove bullet points, numbers at start
    text = re.sub(r'^[\d\.\)\-•∙◦▪▫]+\s*', '', text)
    # Ensure proper ending
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text

def split_into_points(text):
    """Split text into individual remedy points"""
    if not text or not isinstance(text, str):
        return []
    
    points = []
    
    # Try splitting by newlines first
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) > 1:
        for line in lines:
            cleaned = clean_remedy_text(line)
            if cleaned and len(cleaned) > 15 and is_english(cleaned):
                points.append(cleaned)
        if points:
            return points
    
    # Try splitting by periods for long text
    if len(text) > 200:
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 20:
                cleaned = clean_remedy_text(sentence)
                if cleaned and is_english(cleaned):
                    points.append(cleaned)
        if points:
            return points[:10]  # Limit to 10 points
    
    # If no splitting worked, return as single point if valid
    cleaned = clean_remedy_text(text)
    if cleaned and is_english(cleaned):
        return [cleaned]
    
    return []

def fetch_google_results(disease_name, num_results=5):
    """Fetch English-only results from Google Custom Search"""
    query = f"{disease_name} home remedy"
    url = f"https://www.googleapis.com/customsearch/v1?q={quote_plus(query)}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&num={num_results}&lr=lang_en"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        results = []
        for item in data.get("items", []):
            snippet = item.get("snippet", "")
            # Only include English snippets
            if is_english(snippet):
                results.append({
                    "title": item.get("title"),
                    "snippet": snippet,
                    "link": item.get("link")
                })
        return results
    except Exception as e:
        return []

def extract_remedy_from_google(google_results):
    """Extract home remedy points from Google search results (English only)"""
    if not google_results:
        return ["No remedies found online. Please consult a healthcare professional."]
    
    remedies = []
    for result in google_results[:5]:
        snippet = result.get('snippet', '').strip()
        if snippet and is_english(snippet):
            # Split long snippets into sentences
            if len(snippet) > 150:
                sentences = snippet.replace('...', '.').split('. ')
                for sentence in sentences:
                    cleaned = clean_remedy_text(sentence)
                    if cleaned and len(cleaned) > 30 and is_english(cleaned):
                        remedies.append(cleaned)
                        if len(remedies) >= 10:
                            break
            else:
                cleaned = clean_remedy_text(snippet)
                if cleaned and is_english(cleaned):
                    remedies.append(cleaned)
        
        if len(remedies) >= 10:
            break
    
    if remedies:
        return remedies[:10]
    else:
        return ["No specific English remedies found. Please consult a healthcare professional."]

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
        remedy_points = split_into_points(match["Home Remedy"])
        
        return {
            "Item": match["Name of Item"],
            "Disease": match["Disease"].title(),
            "HomeRemedy": remedy_points if remedy_points else ["No remedy details available"],
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
        remedy_points = split_into_points(match["Home Remedy"])
        
        return {
            "Item": match["Name of Item"],
            "Disease": match["Disease"].title(),
            "HomeRemedy": remedy_points if remedy_points else ["No remedy details available"],
            "Yogasan": match["Yogasan"],
            "Image": match["Image"],
            "Link": match["Link"],
            "Source": f"Database (Cosine {confidence:.2%})",
            "Confidence": f"{confidence:.2%}"
        }
    
    # Fallback to Google Custom Search (English only)
    google_results = fetch_google_results(disease_name)
    extracted_remedies = extract_remedy_from_google(google_results)
    
    return {
        "Item": "Web Search",
        "Disease": disease_name.title(),
        "HomeRemedy": extracted_remedies,
        "Yogasan": "Not available",
        "Image": "https://via.placeholder.com/150",
        "Link": google_results[0]["link"] if google_results else "",
        "Source": "Google Search Results (English)",
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
        remedy_points = split_into_points(str(match["Home Remedy"]))
        
        results.append({
            "Item": str(match["Name of Item"]),
            "Disease": str(match["Disease"]).title(),
            "HomeRemedy": remedy_points if remedy_points else ["No remedy details available"],
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
    test_diseases = ["cold", "fever", "piles", "unknown disease xyz"]
    
    for disease in test_diseases:
        result = predict_home_remedy(disease)
        print(f"\n{'='*60}")
        print(f"🔍 Disease: {disease}")
        print(f"📊 Source: {result['Source']}")
        print(f"💊 Item: {result['Item']}")
        print(f"🎯 Confidence: {result['Confidence']}")
        print(f"\n📋 Home Remedies (Point-wise):")
        
        if isinstance(result['HomeRemedy'], list):
            for i, remedy in enumerate(result['HomeRemedy'], 1):
                print(f"  {i}. {remedy}")
        else:
            print(f"  {result['HomeRemedy']}")
        
        print(f"\n🧘 Yogasan: {result['Yogasan']}")
        print(f"🔗 Link: {result['Link']}")
        
        if "GoogleResults" in result and result["GoogleResults"]:
            print(f"\n🌐 Google Sources ({len(result['GoogleResults'])} results):")
            for i, g in enumerate(result["GoogleResults"][:3], 1):
                print(f"  {i}. {g['title']}")
                print(f"     {g['link']}")
    
    print(f"\n{'='*60}")
    print("✅ Test completed!")