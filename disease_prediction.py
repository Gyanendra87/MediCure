import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from fastapi import APIRouter, HTTPException, Query
from sklearn.metrics import accuracy_score

# Create router
health_router = APIRouter(prefix="/health", tags=["Health"])

# ===================================================
# Load Symptoms Dataset
# ===================================================
try:
    symptoms_df = pd.read_csv("symtoms_df.csv")
except Exception as e:
    raise RuntimeError(f"Could not load symtoms_df.csv: {e}")

symptom_cols = [col for col in symptoms_df.columns if col.lower().startswith("symptom")]
for col in symptom_cols:
    symptoms_df[col] = symptoms_df[col].fillna("")

symptoms_df["all_symptoms"] = symptoms_df[symptom_cols].agg(" ".join, axis=1)

X = symptoms_df["all_symptoms"]
y = symptoms_df["Disease"].astype(str)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

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

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2f}")

# ===================================================
# Load Helper Tables
# ===================================================
def load_csv_safe(path, required_columns=None):
    try:
        df = pd.read_csv(path)
        return df
    except:
        return pd.DataFrame(columns=required_columns if required_columns else [])

medications_df = load_csv_safe("medications.csv")
diet_df = load_csv_safe("diets.csv")
description_df = load_csv_safe("descriptions.csv")
workout_df = load_csv_safe("workout.csv")

# ===================================================
# Helper Functions
# ===================================================
def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    text = re.sub(r'\bnan\b|\bNaN\b|\bNone\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\[\]\{\}\'\"]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_value(df, disease, column):
    if df.empty:
        return None
    df[df.columns[0]] = df[df.columns[0]].astype(str)
    match = df[df.iloc[:, 0].str.lower() == disease.lower()]
    if not match.empty and column in match.columns:
        value = match.iloc[0][column]
        return clean_text(str(value))
    return None

# ===================================================
# Predict Endpoint
# ===================================================
@health_router.post("/predict_disease")
async def predict_disease(symptoms: str = Query(...)):
    if not symptoms:
        raise HTTPException(400, "Symptoms required")

    vector = vectorizer.transform([symptoms])
    pred = clf.predict(vector)[0]
    disease = label_encoder.inverse_transform([pred])[0]
    disease = clean_text(disease)

    description = get_value(description_df, disease, "Description")
    medication = get_value(medications_df, disease, "Medication")
    diet = get_value(diet_df, disease, "Diet")
    workout = get_value(workout_df, disease, "workout")

    return {
        "predicted_disease": disease,
        "description": description,
        "medications": medication,
        "diet": diet,
        "workout": workout
    }

@health_router.get("/")
def info():
    return {
        "message": "Health module running",
        "accuracy": accuracy
    }
