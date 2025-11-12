from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Load your saved model ---
model = load_model('xray_tabular_cnn_model.keras')

# --- Load your MultiLabelBinarizer ---
with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)

# --- Image preprocessing ---
def preprocess_image(image_path, target_size=(128, 128)):
    try:
        img = load_img(image_path, target_size=target_size)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Prediction function ---
def predict_findings(image_path, model, mlb, threshold=0.5):
    X_img_input = preprocess_image(image_path)
    if X_img_input is None:
        return []

    # --- Create dummy tabular input ---
    tabular_input_dim = model.inputs[1].shape[1]  # automatically match model
    tabular_input = np.zeros((1, tabular_input_dim))

    # --- Predict ---
    pred = model.predict([X_img_input, tabular_input])
    print("Raw model output:", pred)

    pred_binary = (pred > threshold).astype(int)
    predicted_labels = mlb.inverse_transform(pred_binary)

    # Fallback to top prediction if no labels above threshold
    if not predicted_labels or not predicted_labels[0]:
        top_idx = np.argmax(pred)
        predicted_labels = [mlb.classes_[top_idx]]

    return predicted_labels[0]

# --- Run prediction ---
image_path = r"images\images\00001330_002.png"  # Make sure this image exists
predicted_findings = predict_findings(image_path, model, mlb)
print("Predicted findings:", predicted_findings)
