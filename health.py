import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from fastapi import APIRouter, HTTPException, Query

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

# Vectorize symptoms
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train XGBoost classifier
clf = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=42)
clf.fit(X_vec, y_encoded)

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
# Helper: Safe data fetcher
# ===================================================
def get_value(df, disease_name, column):
    if df.empty:
        return f"No {column} data available."
    df[df.columns[0]] = df[df.columns[0]].astype(str)
    match = df[df.iloc[:, 0].str.lower() == str(disease_name).lower()]
    if not match.empty and column in match.columns:
        val = match.iloc[0][column]
        if isinstance(val, str) and val.strip():
            try:
                return eval(val) if val.strip().startswith("[") else val
            except:
                return val
    return f"No {column} available for this disease."

# ===================================================
# Helper: Get Home Remedies + Yogasan
# ===================================================
def get_remedies(disease_name: str):
    if remedies_df.empty:
        return {"home_remedy": ["No home remedies found"], "yogasan": []}

    df = remedies_df[remedies_df["Disease"].str.lower() == disease_name.lower()]
    if df.empty:
        return {"home_remedy": ["No home remedies found"], "yogasan": []}

    remedies_list = [f"{row['Name of Item']}: {row['Home Remedy']}" for _, row in df.iterrows()]
    yogasan_list = [row["Yogasan"] for _, row in df.iterrows() if pd.notna(row.get("Yogasan"))]

    return {"home_remedy": remedies_list, "yogasan": yogasan_list}

# ===================================================
# Endpoint: Predict Disease from Symptoms
# ===================================================
@health_router.post("/predict_disease")
def predict_disease(symptoms: str = Query(..., description="Comma separated list of symptoms")):
    if not symptoms.strip():
        raise HTTPException(status_code=400, detail="Symptoms input cannot be empty.")

    try:
        X_test = vectorizer.transform([symptoms])
        pred_encoded = clf.predict(X_test)[0]
        disease_pred = label_encoder.inverse_transform([pred_encoded])[0]

        description = get_value(descriptions_df, disease_pred, "Description")
        medications = get_value(medications_df, disease_pred, "Medication")
        diet = get_value(diets_df, disease_pred, "Diet")
        workouts = get_value(workout_df, disease_pred, "workout")
        remedies = get_remedies(disease_pred)

        return {
            "predicted_disease": disease_pred,
            "description": description,
            "medications": medications,
            "diet": diet,
            "workout": workouts,
            "home_remedies": remedies["home_remedy"],
            "yogasan": remedies["yogasan"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ===================================================
# Endpoint: Get Home Remedies for any disease (direct lookup)
# ===================================================
@health_router.get("/get_remedies")
def get_remedies_endpoint(disease: str = Query(..., description="Disease name to fetch remedies for")):
    if not disease.strip():
        raise HTTPException(status_code=400, detail="Disease name cannot be empty.")
    remedies = get_remedies(disease)
    return {"disease": disease, "home_remedies": remedies["home_remedy"], "yogasan": remedies["yogasan"]}

# ===================================================
# Endpoint: Frontend-compatible predict_remedy
# ===================================================
@health_router.get("/predict_remedy")
def predict_remedy_endpoint(disease: str = Query(..., description="Disease name to fetch remedies for frontend")):
    if not disease.strip():
        raise HTTPException(status_code=400, detail="Disease name cannot be empty.")
    remedies = get_remedies(disease)
    return {"disease": disease, "home_remedies": remedies["home_remedy"], "yogasan": remedies["yogasan"]}

# ===================================================
# Health check
# ===================================================
@health_router.get("/")
def health_root():
    return {"message": "✅ Health + Home Remedies prediction module running (XGBoost)!"}
