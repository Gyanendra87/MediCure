import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, 
                                      Concatenate, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import DenseNet121
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pickle

# === Paths ===
IMAGE_DIR = r"images\images"
CSV_PATH = "labels.csv"

# === Load CSV ===
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Drop rows with missing images
df["Image Index"] = df["Image Index"].apply(lambda x: os.path.join(IMAGE_DIR, x))
df = df[df["Image Index"].apply(os.path.exists)]
print(f"✅ Loaded {len(df)} samples")

# === Tabular features ===
tabular_cols = ["Patient Age", "Patient Sex", "View Position"]
X_tab = df[tabular_cols].copy()

# Encode categorical columns
le_dict = {}
for col in X_tab.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_tab[col] = le.fit_transform(X_tab[col])
    le_dict[col] = le

# Scale numeric columns
num_cols = X_tab.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_tab[num_cols] = scaler.fit_transform(X_tab[num_cols])

# === Multi-label target ===
y_labels = df["Finding Labels"].apply(lambda x: x.split('|'))
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y_labels)
num_classes = y.shape[1]
print(f"✅ Found {num_classes} unique labels: {mlb.classes_}")

# === Preprocess images with data augmentation ===
IMG_SIZE = (224, 224)  # Larger size for better feature extraction

def preprocess_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    return img_array

X_img = np.array([preprocess_image(path) for path in df["Image Index"]])

# === Split data with stratification ===
X_img_train, X_img_val, X_tab_train, X_tab_val, y_train, y_val = train_test_split(
    X_img, X_tab.values, y, test_size=0.2, random_state=42, stratify=y[:, 0]
)

# === Data Augmentation ===
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)

# === Build Enhanced Model with Transfer Learning ===
# Image branch using DenseNet121 (pre-trained on ImageNet)
img_input = Input(shape=(224, 224, 3))
base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=img_input)

# Fine-tune last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)

# Tabular branch with deeper architecture
tab_input = Input(shape=(X_tab_train.shape[1],))
t = Dense(128, activation='relu')(tab_input)
t = BatchNormalization()(t)
t = Dropout(0.4)(t)
t = Dense(64, activation='relu')(t)
t = BatchNormalization()(t)
t = Dropout(0.3)(t)

# Combine branches
combined = Concatenate()([x, t])
z = Dense(256, activation='relu')(combined)
z = BatchNormalization()(z)
z = Dropout(0.5)(z)
z = Dense(128, activation='relu')(z)
z = Dropout(0.4)(z)
output = Dense(num_classes, activation='sigmoid')(z)

# === Compile Model ===
model = Model(inputs=[img_input, tab_input], outputs=output)

# Use Adam with custom learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()

# === Callbacks ===
callbacks = [
    EarlyStopping(
        monitor='val_auc',
        patience=10,
        restore_best_weights=True,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_xray_model.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# === Class weights for imbalanced data ===
class_weights = {}
for i in range(num_classes):
    classes = np.unique(y_train[:, i])
    if len(classes) > 1:
        weights = compute_class_weight('balanced', classes=classes, y=y_train[:, i])
        class_weights[i] = dict(zip(classes, weights))

# === Train with data augmentation ===
def data_generator(X_img, X_tab, y, batch_size=32, augment=False):
    num_samples = len(X_img)
    while True:
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            
            X_img_batch = X_img[batch_indices]
            X_tab_batch = X_tab[batch_indices]
            y_batch = y[batch_indices]
            
            if augment:
                X_img_batch = datagen.flow(X_img_batch, batch_size=len(X_img_batch), shuffle=False).next()
            
            yield [X_img_batch, X_tab_batch], y_batch

# Calculate steps
batch_size = 16  # Smaller batch size for better generalization
steps_per_epoch = len(X_img_train) // batch_size
validation_steps = len(X_img_val) // batch_size

print("\n🚀 Starting training with enhanced architecture...")
print(f"Training samples: {len(X_img_train)}, Validation samples: {len(X_img_val)}")

history = model.fit(
    data_generator(X_img_train, X_tab_train, y_train, batch_size, augment=True),
    steps_per_epoch=steps_per_epoch,
    validation_data=data_generator(X_img_val, X_tab_val, y_val, batch_size, augment=False),
    validation_steps=validation_steps,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# === Fine-tuning phase ===
print("\n🔧 Fine-tuning: Unfreezing all layers...")
for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

history_fine = model.fit(
    data_generator(X_img_train, X_tab_train, y_train, batch_size, augment=True),
    steps_per_epoch=steps_per_epoch,
    validation_data=data_generator(X_img_val, X_tab_val, y_val, batch_size, augment=False),
    validation_steps=validation_steps,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

# === Save model and encoders ===
model.save("xray_tabular_cnn_model_enhanced.keras")
pickle.dump(le_dict, open("label_encoders.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(mlb, open("mlb.pkl", "wb"))

# === Evaluation ===
print("\n📊 Final Evaluation:")
val_loss, val_acc, val_auc, val_precision, val_recall = model.evaluate(
    [X_img_val, X_tab_val], y_val, verbose=0
)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
print(f"Validation AUC: {val_auc:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")

print("\n✅ Enhanced model and encoders saved successfully")