import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from fastapi import APIRouter, HTTPException, Query
from sklearn.metrics import accuracy_score
import numpy as np


health_router = APIRouter(prefix="/health", tags=["Health"])


# Load Symptoms Dataset

try:
    symptoms_df = pd.read_csv("symtoms_df.csv")
    print(f"Loaded dataset with {len(symptoms_df)} records")
except Exception as e:
    raise RuntimeError(f"Could not load symtoms_df.csv: {e}")


symptoms_df["Disease"] = symptoms_df["Disease"].astype(str).str.strip()


symptom_cols = [col for col in symptoms_df.columns if col.lower().startswith("symptom")]
print(f"Found {len(symptom_cols)} symptom columns")


for col in symptom_cols:
    symptoms_df[col] = symptoms_df[col].fillna("").astype(str).str.strip().str.lower()


symptoms_df["all_symptoms"] = symptoms_df[symptom_cols].apply(
    lambda row: " ".join([s for s in row if s and s != "nan" and s != ""]), 
    axis=1
)

print(f"Sample symptoms from training data: {symptoms_df['all_symptoms'].iloc[0][:100]}")


symptoms_df = symptoms_df[symptoms_df["all_symptoms"].str.len() > 0].reset_index(drop=True)
print(f"After cleaning: {len(symptoms_df)} records with valid symptoms")

X = symptoms_df["all_symptoms"]
y = symptoms_df["Disease"]


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Number of unique diseases: {len(label_encoder.classes_)}")


vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    lowercase=True,
    min_df=1,  
    max_df=0.95  
)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


clf = XGBClassifier(
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    n_estimators=300,  
    max_depth=10, 
    learning_rate=0.05, 
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.1
)

print("Training model...")
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")


def load_csv_safe(path, required_columns=None):
    try:
        df = pd.read_csv(path)
        
    
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
workout_df = load_csv_safe("workout_df.csv") 


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

def clean_text(text):
    """Clean text by removing NaN, None, and extra formatting"""
    if pd.isna(text) or text is None:
        return None
    text = str(text)
   
    text = re.sub(r'\bnan\b|\bNaN\b|\bNone\b', '', text, flags=re.IGNORECASE)
 
    text = re.sub(r'[\[\]\{\}\'\"]+', '', text)

    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text if text else None

def get_value(df, disease, column):
    """Get value from dataframe for a specific disease"""
    if df.empty or disease is None:
        print(f"Empty dataframe or no disease for column: {column}")
        return None
    
  
    df_copy = df.copy()
    
 
    disease_col = None
    for col in df_copy.columns:
        if 'disease' in col.lower():
            disease_col = col
            break
    
 
    if disease_col is None:
        disease_col = df_copy.columns[0]
    
   
    df_copy[disease_col] = df_copy[disease_col].astype(str).str.strip()
    
   
    match = df_copy[df_copy[disease_col].str.lower() == disease.lower()]
    
 
    if match.empty:
        match = df_copy[df_copy[disease_col].str.lower().str.contains(disease.lower(), na=False, regex=False)]
    
    if match.empty:
        disease_normalized = disease.replace("_", " ").replace("-", " ")
        df_copy['normalized'] = df_copy[disease_col].str.lower().str.replace("_", " ").str.replace("-", " ")
        match = df_copy[df_copy['normalized'] == disease_normalized.lower()]
    
    if not match.empty:
       
        possible_columns = [
            column,  
            column.lower(), 
            column.capitalize(), 
            column.title(), 
            column.upper()  
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
    
    
    symptoms = symptoms.lower().strip()
    
    symptoms = symptoms.replace(' ', '_')
    

    symptoms = re.sub(r'[^\w\s_]', '_', symptoms)
    
    
    symptoms = re.sub(r'_+', '_', symptoms)
    
    symptoms = symptoms.strip('_')
    
    return symptoms

@health_router.post("/predict_disease")
async def predict_disease(symptoms: str = Query(...)):
    """
    Predict disease based on symptoms.
    
    Example symptoms: "fever headache cough"
    """
    if not symptoms or not symptoms.strip():
        raise HTTPException(400, "Symptoms required")

    try:
       
        processed_symptoms = preprocess_symptoms(symptoms)
        
        if not processed_symptoms:
            raise HTTPException(400, "No valid symptoms provided after processing")
        
        vector = vectorizer.transform([processed_symptoms])
        pred_encoded = clf.predict(vector)[0]
        pred_proba = clf.predict_proba(vector)[0]
   
        disease = label_encoder.inverse_transform([pred_encoded])[0]
        disease = disease.strip()
        
     
        confidence = float(pred_proba[pred_encoded])
        
 
        top_3_idx = np.argsort(pred_proba)[-3:][::-1]
        top_3_diseases = [
            {
                "disease": label_encoder.inverse_transform([idx])[0].strip(),
                "confidence": float(pred_proba[idx])
            }
            for idx in top_3_idx
        ]
        

        description = get_value(description_df, disease, "Description")
        medication = get_value(medications_df, disease, "Medication")
      
        precaution_cols = ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
        precautions = []
        if not precautions_df.empty:
            precautions_df_copy = precautions_df.copy()
      
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
        
   
        diet = None
        for diet_col in ["Diet", "diet", "Dietary Recommendations", "Recommended Diet"]:
            diet = get_value(diet_df, disease, diet_col)
            if diet:
                break
        
        workout = None
        if not workout_df.empty:
            workout_df_copy = workout_df.copy()
            
        
            disease_col = None
            for col in workout_df_copy.columns:
                if 'disease' in col.lower():
                    disease_col = col
                    break
            
            if disease_col is None:
                disease_col = workout_df_copy.columns[0]
            
            workout_df_copy[disease_col] = workout_df_copy[disease_col].astype(str).str.strip()
            
         
            match = workout_df_copy[workout_df_copy[disease_col].str.lower() == disease.lower()]
            
           
            if match.empty:
                disease_normalized = disease.replace("_", " ").replace("-", " ")
                workout_df_copy['normalized'] = workout_df_copy[disease_col].str.lower().str.replace("_", " ").str.replace("-", " ")
                match = workout_df_copy[workout_df_copy['normalized'] == disease_normalized.lower()]
            
            
            if match.empty:
                match = workout_df_copy[workout_df_copy[disease_col].str.lower().str.contains(disease.lower(), na=False, regex=False)]
            
            if not match.empty:
              
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