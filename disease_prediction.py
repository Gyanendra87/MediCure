import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from fastapi import APIRouter, HTTPException, Query
from sklearn.metrics import accuracy_score
import numpy as np

# Create router
health_router = APIRouter(prefix="/health", tags=["Health"])

# ===================================================
# Load Symptoms Dataset
# ===================================================
try:
    symptoms_df = pd.read_csv("symtoms_df.csv")
    print(f"Loaded dataset with {len(symptoms_df)} records")
except Exception as e:
    raise RuntimeError(f"Could not load symtoms_df.csv: {e}")

# Clean disease names first
symptoms_df["Disease"] = symptoms_df["Disease"].astype(str).str.strip()

# Get symptom columns
symptom_cols = [col for col in symptoms_df.columns if col.lower().startswith("symptom")]
print(f"Found {len(symptom_cols)} symptom columns")

# Fill NaN and clean symptoms - keep underscores
for col in symptom_cols:
    symptoms_df[col] = symptoms_df[col].fillna("").astype(str).str.strip().str.lower()

# Combine all symptoms - preserve underscores by using space separator
symptoms_df["all_symptoms"] = symptoms_df[symptom_cols].apply(
    lambda row: " ".join([s for s in row if s and s != "nan" and s != ""]), 
    axis=1
)

print(f"Sample symptoms from training data: {symptoms_df['all_symptoms'].iloc[0][:100]}")

# Remove rows with no symptoms
symptoms_df = symptoms_df[symptoms_df["all_symptoms"].str.len() > 0].reset_index(drop=True)
print(f"After cleaning: {len(symptoms_df)} records with valid symptoms")

X = symptoms_df["all_symptoms"]
y = symptoms_df["Disease"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Number of unique diseases: {len(label_encoder.classes_)}")

# Vectorize with better parameters
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    lowercase=True,
    min_df=1,  # Include even rare symptoms
    max_df=0.95  # Remove very common words
)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train model with better parameters
clf = XGBClassifier(
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    n_estimators=300,  # Increased
    max_depth=10,  # Increased
    learning_rate=0.05,  # Decreased for better learning
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.1
)

print("Training model...")
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# ===================================================
# Load Helper Tables
# ===================================================
def load_csv_safe(path, required_columns=None):
    try:
        df = pd.read_csv(path)
        
        # Drop any unnamed index columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            print(f"Dropped unnamed columns from {path}: {unnamed_cols}")
        
        print(f"Loaded {path}: {len(df)} records")
        return df
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return pd.DataFrame(columns=required_columns if required_columns else [])

medications_df = load_csv_safe("medications.csv")
diet_df = load_csv_safe("diets.csv")
description_df = load_csv_safe("descriptions.csv")
precautions_df = load_csv_safe("precautions.csv")
workout_df = load_csv_safe("workout_df.csv")  # Changed from workout.csv

# Print available columns for debugging
if not workout_df.empty:
    print(f"Workout columns: {workout_df.columns.tolist()}")
if not medications_df.empty:
    print(f"Medications columns: {medications_df.columns.tolist()}")
if not diet_df.empty:
    print(f"Diet columns: {diet_df.columns.tolist()}")
if not description_df.empty:
    print(f"Description columns: {description_df.columns.tolist()}")
if not precautions_df.empty:
    print(f"Precautions columns: {precautions_df.columns.tolist()}")

# ===================================================
# Helper Functions
# ===================================================
def clean_text(text):
    """Clean text by removing NaN, None, and extra formatting"""
    if pd.isna(text) or text is None:
        return None
    text = str(text)
    # Remove various forms of "nan" or "None"
    text = re.sub(r'\bnan\b|\bNaN\b|\bNone\b', '', text, flags=re.IGNORECASE)
    # Remove brackets and quotes
    text = re.sub(r'[\[\]\{\}\'\"]+', '', text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text if text else None

def get_value(df, disease, column):
    """Get value from dataframe for a specific disease"""
    if df.empty or disease is None:
        print(f"Empty dataframe or no disease for column: {column}")
        return None
    
    # Make a copy and identify the disease column
    df_copy = df.copy()
    
    # Find the disease column (look for 'disease' or 'Disease' in column names)
    disease_col = None
    for col in df_copy.columns:
        if 'disease' in col.lower():
            disease_col = col
            break
    
    # If no disease column found, assume first column
    if disease_col is None:
        disease_col = df_copy.columns[0]
    
    # Clean and prepare disease column for matching
    df_copy[disease_col] = df_copy[disease_col].astype(str).str.strip()
    
    # Try exact match first (case-insensitive)
    match = df_copy[df_copy[disease_col].str.lower() == disease.lower()]
    
    # If no match, try partial match
    if match.empty:
        match = df_copy[df_copy[disease_col].str.lower().str.contains(disease.lower(), na=False, regex=False)]
    
    # If still no match, try with underscores replaced
    if match.empty:
        disease_normalized = disease.replace("_", " ").replace("-", " ")
        df_copy['normalized'] = df_copy[disease_col].str.lower().str.replace("_", " ").str.replace("-", " ")
        match = df_copy[df_copy['normalized'] == disease_normalized.lower()]
    
    if not match.empty:
        # Check all possible column name variations
        possible_columns = [
            column,  # Original column name
            column.lower(),  # Lowercase
            column.capitalize(),  # Capitalized
            column.title(),  # Title case
            column.upper()  # Uppercase
        ]
        
        for col in possible_columns:
            if col in match.columns:
                value = match.iloc[0][col]
                cleaned = clean_text(value)
                if cleaned:
                    print(f"Found value for {disease} in column {col}: {cleaned[:50]}...")
                    return cleaned
                else:
                    print(f"Empty value for {disease} in column {col}")
        
        print(f"Column '{column}' not found. Available columns: {match.columns.tolist()}")
        return None
    
    print(f"No match found for disease: {disease}")
    return None

def preprocess_symptoms(symptoms: str) -> str:
    """Preprocess symptoms to match training data format"""
    if not symptoms:
        return ""
    
    # Convert to lowercase
    symptoms = symptoms.lower().strip()
    
    # Replace spaces with underscores to match CSV format (e.g., "chest pain" -> "chest_pain")
    symptoms = symptoms.replace(' ', '_')
    
    # Remove extra punctuation except underscores
    symptoms = re.sub(r'[^\w\s_]', '_', symptoms)
    
    # Remove extra underscores
    symptoms = re.sub(r'_+', '_', symptoms)
    
    # Remove leading/trailing underscores
    symptoms = symptoms.strip('_')
    
    return symptoms

# ===================================================
# Predict Endpoint
# ===================================================
@health_router.post("/predict_disease")
async def predict_disease(symptoms: str = Query(...)):
    """
    Predict disease based on symptoms.
    
    Example symptoms: "fever headache cough"
    """
    if not symptoms or not symptoms.strip():
        raise HTTPException(400, "Symptoms required")

    try:
        # Preprocess symptoms
        processed_symptoms = preprocess_symptoms(symptoms)
        
        if not processed_symptoms:
            raise HTTPException(400, "No valid symptoms provided after processing")
        
        # Vectorize and predict
        vector = vectorizer.transform([processed_symptoms])
        pred_encoded = clf.predict(vector)[0]
        pred_proba = clf.predict_proba(vector)[0]
        
        # Get predicted disease
        disease = label_encoder.inverse_transform([pred_encoded])[0]
        disease = disease.strip()
        
        # Get confidence
        confidence = float(pred_proba[pred_encoded])
        
        # Get top 3 predictions
        top_3_idx = np.argsort(pred_proba)[-3:][::-1]
        top_3_diseases = [
            {
                "disease": label_encoder.inverse_transform([idx])[0].strip(),
                "confidence": float(pred_proba[idx])
            }
            for idx in top_3_idx
        ]
        
        # Get additional information
        description = get_value(description_df, disease, "Description")
        medication = get_value(medications_df, disease, "Medication")
        
        # Get precautions
        precaution_cols = ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
        precautions = []
        if not precautions_df.empty:
            precautions_df_copy = precautions_df.copy()
            
            # Find disease column
            disease_col = None
            for col in precautions_df_copy.columns:
                if 'disease' in col.lower():
                    disease_col = col
                    break
            
            if disease_col:
                precautions_df_copy[disease_col] = precautions_df_copy[disease_col].astype(str).str.strip()
                match = precautions_df_copy[precautions_df_copy[disease_col].str.lower() == disease.lower()]
                
                if not match.empty:
                    for col in precaution_cols:
                        if col in match.columns:
                            val = clean_text(match.iloc[0][col])
                            if val:
                                precautions.append(val)
        
        # Try different possible column names for diet
        diet = None
        for diet_col in ["Diet", "diet", "Dietary Recommendations", "Recommended Diet"]:
            diet = get_value(diet_df, disease, diet_col)
            if diet:
                break
        
        # Try different possible column names for workout with better matching
        workout = None
        if not workout_df.empty:
            workout_df_copy = workout_df.copy()
            
            # Find the disease column
            disease_col = None
            for col in workout_df_copy.columns:
                if 'disease' in col.lower():
                    disease_col = col
                    break
            
            if disease_col is None:
                disease_col = workout_df_copy.columns[0]
            
            workout_df_copy[disease_col] = workout_df_copy[disease_col].astype(str).str.strip()
            
            # Try exact match
            match = workout_df_copy[workout_df_copy[disease_col].str.lower() == disease.lower()]
            
            # Try with spaces instead of underscores
            if match.empty:
                disease_normalized = disease.replace("_", " ").replace("-", " ")
                workout_df_copy['normalized'] = workout_df_copy[disease_col].str.lower().str.replace("_", " ").str.replace("-", " ")
                match = workout_df_copy[workout_df_copy['normalized'] == disease_normalized.lower()]
            
            # Try partial match
            if match.empty:
                match = workout_df_copy[workout_df_copy[disease_col].str.lower().str.contains(disease.lower(), na=False, regex=False)]
            
            if not match.empty:
                # Try different column names
                for workout_col in ["workout", "Workout", "Exercise", "exercise", "Physical Activity", "Recommended Exercise"]:
                    if workout_col in match.columns:
                        workout = clean_text(match.iloc[0][workout_col])
                        if workout:
                            print(f"Found workout for {disease}: {workout[:50]}...")
                            break
                
                if not workout:
                    print(f"No workout value found. Available columns: {match.columns.tolist()}")
            else:
                print(f"No match found in workout_df for disease: {disease}")
        
        return {
            "predicted_disease": disease,
            "confidence": round(confidence, 4),
            "model_accuracy": round(accuracy, 4),
            "top_predictions": top_3_diseases,
            "description": description,
            "medications": medication,
            "precautions": precautions if precautions else None,
            "diet": diet,
            "workout": workout,
            "processed_symptoms": processed_symptoms
        }
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise HTTPException(500, f"Prediction error: {str(e)}")

@health_router.get("/")
def info():
    """Get API information and model statistics"""
    return {
        "message": "Health module running",
        "accuracy": round(accuracy, 4),
        "total_diseases": len(label_encoder.classes_),
        "total_samples": len(symptoms_df),
        "model_type": "XGBoost Classifier"
    }

@health_router.get("/diseases")
def list_diseases():
    """List all available diseases in the model"""
    return {
        "diseases": sorted(label_encoder.classes_.tolist()),
        "total": len(label_encoder.classes_)
    }