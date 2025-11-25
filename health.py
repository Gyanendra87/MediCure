import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from fastapi import APIRouter, HTTPException, Query
import httpx
from bs4 import BeautifulSoup

# Initialize router
health_router = APIRouter(prefix="/health", tags=["Health"])

# ===================================================
# Load and prepare symptoms dataset for XGBoost
# ===================================================
try:
    symptoms_df = pd.read_csv("symtoms_df.csv")
except Exception as e:
    raise RuntimeError(f"Could not load symtoms_df.csv: {e}")

# Combine all symptom columns into one text column
symptom_cols = [col for col in symptoms_df.columns if col.lower().startswith("symptom")]
for col in symptom_cols:
    symptoms_df[col] = symptoms_df[col].fillna("")

symptoms_df["all_symptoms"] = symptoms_df[symptom_cols].agg(" ".join, axis=1)

X = symptoms_df["all_symptoms"]
y = symptoms_df["Disease"].astype(str)

# Encode disease labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Use TfidfVectorizer instead of CountVectorizer for better accuracy
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

# Split into training and testing sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train XGBoost classifier with optimized parameters
clf = XGBClassifier(
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
clf.fit(X_train, y_train)

# Evaluate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# ===================================================
# Load supporting datasets
# ===================================================
def load_csv_safe(path, required_columns=None):
    try:
        df = pd.read_csv(path)
        if required_columns:
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in {path}")
        return df
    except Exception as e:
        print(f"⚠ Could not load {path}: {e}")
        return pd.DataFrame(columns=required_columns if required_columns else [])

medications_df = load_csv_safe("medications.csv", ["Disease", "Medication"])
diets_df = load_csv_safe("diets.csv", ["Disease", "Diet"])
descriptions_df = load_csv_safe("descriptions.csv", ["Disease", "Description"])
workout_df = load_csv_safe("workout.csv", ["disease", "workout"])
remedies_df = load_csv_safe("remedies.csv", ["Name of Item", "Disease", "Home Remedy", "Yogasan"])

# ===================================================
# Helper: Clean and format text
# ===================================================
def clean_text(text):
    """Remove NaN, clean unwanted characters, and format properly"""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    # Remove common data artifacts
    text = re.sub(r'\bnan\b|\bNaN\b|\bNone\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\[\]\{\}\'\"]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def format_as_points(text, max_length=150):
    """Convert long text into bullet points"""
    if not text or len(text) < max_length:
        return text
    
    # Split by common delimiters
    sentences = re.split(r'[.;,]\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    if len(sentences) <= 1:
        return text
    
    # Format as bullet points
    return "\n• " + "\n• ".join(sentences[:8])  # Limit to 8 points

# ===================================================
# Helper: Fetch remedies from internet
# ===================================================
async def fetch_remedies_from_web(disease_name: str):
    """Fetch home remedies from the internet when not in database"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Search query
            query = f"{disease_name} home remedies natural treatment"
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = await client.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract snippets from search results
            remedies = []
            snippets = soup.find_all(['span', 'div'], class_=lambda x: x and 'BNeawe' in x)
            
            for snippet in snippets[:5]:
                text = snippet.get_text().strip()
                if text and len(text) > 30 and disease_name.lower() in text.lower():
                    cleaned = clean_text(text)
                    if cleaned and len(cleaned) > 20:
                        remedies.append(cleaned)
            
            if remedies:
                return {
                    "home_remedy": remedies[:3],  # Limit to top 3
                    "yogasan": [],
                    "source": "External (Internet)"
                }
            
            # Fallback generic advice
            return {
                "home_remedy": [
                    f"Stay hydrated and get adequate rest",
                    f"Maintain a balanced diet rich in fruits and vegetables",
                    f"Consult a healthcare professional for proper diagnosis"
                ],
                "yogasan": [],
                "source": "External (General Advice)"
            }
            
    except Exception as e:
        print(f"Error fetching from web: {e}")
        return None

# ===================================================
# Helper: Safe data fetcher
# ===================================================
def get_value(df, disease_name, column):
    if df.empty:
        return None
    
    df[df.columns[0]] = df[df.columns[0]].astype(str)
    match = df[df.iloc[:, 0].str.lower() == str(disease_name).lower()]
    
    if not match.empty and column in match.columns:
        val = match.iloc[0][column]
        if pd.notna(val):
            cleaned_val = clean_text(str(val))
            if cleaned_val:
                try:
                    # Try to evaluate if it's a list
                    if cleaned_val.startswith("["):
                        evaluated = eval(cleaned_val)
                        return [clean_text(item) for item in evaluated if clean_text(item)]
                    return format_as_points(cleaned_val)
                except:
                    return format_as_points(cleaned_val)
    
    return None

# ===================================================
# Helper: Get Home Remedies + Yogasan
# ===================================================
async def get_remedies(disease_name: str):
    """Get remedies from database first, then fallback to internet"""
    
    # First try to get from database
    if not remedies_df.empty:
        df = remedies_df[remedies_df["Disease"].str.lower() == disease_name.lower()]
        if not df.empty:
            remedies_list = []
            yogasan_list = []
            
            for _, row in df.iterrows():
                item_name = clean_text(row.get('Name of Item', ''))
                remedy = clean_text(row.get('Home Remedy', ''))
                if item_name and remedy:
                    remedies_list.append(f"{item_name}: {remedy}")
                
                yogasan = clean_text(row.get('Yogasan', ''))
                if yogasan:
                    yogasan_list.append(yogasan)
            
            if remedies_list:
                return {
                    "home_remedy": remedies_list,
                    "yogasan": yogasan_list,
                    "source": "Database"
                }
    
    # If not found in database, fetch from internet
    print(f"⚠ No database remedies for '{disease_name}', fetching from internet...")
    web_remedies = await fetch_remedies_from_web(disease_name)
    
    if web_remedies:
        return web_remedies
    
    # Final fallback
    return {
        "home_remedy": [
            "No specific remedies found. Please consult a healthcare professional.",
            "Maintain good hygiene and healthy lifestyle habits.",
            "Stay hydrated and get adequate rest."
        ],
        "yogasan": [],
        "source": "External (Fallback)"
    }

# ===================================================
# Endpoint: Predict Disease from Symptoms
# ===================================================
@health_router.post("/predict_disease")
async def predict_disease(symptoms: str = Query(..., description="Comma separated list of symptoms")):
    if not symptoms.strip():
        raise HTTPException(status_code=400, detail="Symptoms input cannot be empty.")

    try:
        X_test = vectorizer.transform([symptoms])
        pred_encoded = clf.predict(X_test)[0]
        disease_pred = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Clean disease name
        disease_pred = clean_text(disease_pred)

        # Get all information
        description = get_value(descriptions_df, disease_pred, "Description")
        medications = get_value(medications_df, disease_pred, "Medication")
        diet = get_value(diets_df, disease_pred, "Diet")
        workouts = get_value(workout_df, disease_pred, "workout")
        remedies = await get_remedies(disease_pred)

        return {
            "predicted_disease": disease_pred,
            "description": description or "No description available.",
            "medications": medications or "No medication information available.",
            "diet": diet or "No diet information available.",
            "workout": workouts or "No workout information available.",
            "home_remedies": remedies["home_remedy"],
            "yogasan": remedies["yogasan"],
            "remedy_source": remedies.get("source", "Unknown")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ===================================================
# Endpoint: Get Home Remedies for any disease
# ===================================================
@health_router.get("/get_remedies")
async def get_remedies_endpoint(disease: str = Query(..., description="Disease name to fetch remedies for")):
    if not disease.strip():
        raise HTTPException(status_code=400, detail="Disease name cannot be empty.")
    
    disease = clean_text(disease)
    remedies = await get_remedies(disease)
    
    return {
        "disease": disease,
        "home_remedies": remedies["home_remedy"],
        "yogasan": remedies["yogasan"],
        "source": remedies.get("source", "Unknown")
    }

# ===================================================
# Health check
# ===================================================
@health_router.get("/")
def health_root():
    return {
        "message": "✅ Health + Home Remedies prediction module running (XGBoost)!",
        "model_accuracy": f"{accuracy * 100:.2f}%",
        "training_samples": len(X_train),
        "testing_samples": len(X_test)
    }