import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from fastapi import APIRouter, HTTPException

# Initialize router
health_router = APIRouter(prefix="/health", tags=["Health"])

# ===================================================
# Load and prepare symptoms dataset
# ===================================================
symptoms_df = pd.read_csv("symtoms_df.csv")  # <-- your dataset name

# Combine all symptom columns into one text column
symptom_cols = [col for col in symptoms_df.columns if col.lower().startswith("symptom")]
for col in symptom_cols:
    symptoms_df[col] = symptoms_df[col].fillna("")

symptoms_df["all_symptoms"] = symptoms_df[symptom_cols].agg(" ".join, axis=1)

X = symptoms_df["all_symptoms"]
y = symptoms_df["Disease"].astype(str)  # ensure all are strings

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
def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"⚠ Could not load {path}: {e}")
        return pd.DataFrame()

medications_df = load_csv_safe("medications.csv")   # Disease, Medication
diets_df = load_csv_safe("diets.csv")               # Disease, Diet
descriptions_df = load_csv_safe("descriptions.csv") # Disease, Description
workout_df = load_csv_safe("workout.csv")           # disease, workout
remedies_df = load_csv_safe("remedies.csv")         # Name of Item, Diseases, Home Remedy, Yogasan

# ===================================================
# Helper: Safe data fetcher
# ===================================================
def get_value(df, disease_name, column):
    if df.empty:
        return f"No {column} data available."
    # ensure first column is string
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
def get_remedies(disease_name):
    if remedies_df.empty:
        return {"home_remedy": ["No home remedies found"], "yogasan": []}

    # make sure column exists and is string
    if "Diseases" not in remedies_df.columns:
        return {"home_remedy": ["Invalid remedies CSV format"], "yogasan": []}
    remedies_df["Diseases"] = remedies_df["Diseases"].astype(str)

    df = remedies_df[remedies_df["Diseases"].str.lower() == str(disease_name).lower()]
    if df.empty:
        return {"home_remedy": ["No home remedies found"], "yogasan": []}

    remedies_list = []
    yogasan_list = []
    for _, row in df.iterrows():
        remedies_list.append(f"{row['Name of Item']}: {row['Home Remedy']}")
        if pd.notna(row.get("Yogasan")):
            yogasan_list.append(row["Yogasan"])

    return {"home_remedy": remedies_list, "yogasan": yogasan_list}

# ===================================================
# Endpoint: Predict Disease from Symptoms
# ===================================================
@health_router.post("/predict_disease")
def predict_disease(symptoms: str):
    if not symptoms.strip():
        raise HTTPException(status_code=400, detail="Symptoms input cannot be empty.")

    try:
        # Vectorize and predict
        X_test = vectorizer.transform([symptoms])
        pred_encoded = clf.predict(X_test)[0]
        disease_pred = label_encoder.inverse_transform([pred_encoded])[0]

        # Fetch related information
        description = get_value(descriptions_df, disease_pred, "Description")
        medications = get_value(medications_df, disease_pred, "Medication")
        diet = get_value(diets_df, disease_pred, "Diet")
        workouts = get_value(workout_df, disease_pred, "workout")
        remedies = get_remedies(disease_pred)

        # Final response
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
# Health check
# ===================================================
@health_router.get("/")
def health_root():
    return {"message": "✅ Health prediction module (XGBoost) is running successfully!"}
