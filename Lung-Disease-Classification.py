import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
# ==========================================
# 1. ENVIRONMENT & CONFIGURATION
# ==========================================
DATA_DIR = "nih_dataset/"  # Update with your local dataset path
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5                 # Reduced epochs for faster initial test run
NUM_CLASSES = 14

DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# ==========================================
# 2. DATA PREPROCESSING Pipeline
# ==========================================
def load_and_preprocess_metadata(csv_path):
    df = pd.read_csv(csv_path)
    for label in DISEASE_LABELS:
        df[label] = df['Finding Labels'].apply(lambda x: 1 if label in x else 0)
    return df

def parse_image_function(filename, label):
    """Robust image decoding function using decode_image to prevent formatting errors."""
    image_string = tf.io.read_file(filename)
    # Using decode_image handles various PNG formats cleanly
    image = tf.image.decode_image(image_string, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0  # Normalize to [0,1]
    return image, label

def build_tf_dataset(filenames, labels, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# ==========================================
# 3. ARCHITECTURE DESIGN (Custom CNN)
# ==========================================
def build_lung_cnn_model(input_shape=(224, 224, 3), num_classes=14):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='sigmoid') # Sigmoid for multi-label
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
    )
    return model

# ==========================================
# 4. PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    print("[INFO] Preparing Data Pipeline...")
    
    # Generate cleaner mock dataframe structures
    dummy_data = {
        'Image Index': [f'img_{i}.png' for i in range(100)],
        'Finding Labels': ['Infiltration|Effusion' if i%3==0 else 'Atelectasis' for i in range(100)]
    }
    df = pd.DataFrame(dummy_data)
    for label in DISEASE_LABELS:
        df[label] = df['Finding Labels'].apply(lambda x: 1 if label in x else 0)
    
    # FIXED: Writing clean, standard 8-bit RGB integers to prevent invalid PNG formats
    os.makedirs(DATA_DIR, exist_ok=True)
    for img_name in df['Image Index']:
        img_path = os.path.join(DATA_DIR, img_name)
        # Always overwrite old corrupt placeholder test images
        mock_pixels = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        plt.imsave(img_path, mock_pixels)

    filepaths = [os.path.join(DATA_DIR, name) for name in df['Image Index'].values]
    labels_matrix = df[DISEASE_LABELS].values

    X_train, X_test, y_train, y_test = train_test_split(filepaths, labels_matrix, test_size=0.2, random_state=42)

    train_ds = build_tf_dataset(X_train, y_train, is_training=True)
    test_ds = build_tf_dataset(X_test, y_test, is_training=False)

    print("[INFO] Building and Compiling Deep CNN Architecture...")
    model = build_lung_cnn_model()
    model.summary()

    print("[INFO] Initiating Model Optimization...")
    history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

    # ==========================================
    # 5. METRIC EVALUATION PIPELINE
    # ==========================================
    print("[INFO] Running Multi-Label Pipeline Evaluations...")
    y_pred_probs = model.predict(test_ds)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred_binary, target_names=DISEASE_LABELS, zero_division=0))

    print("\n[SUCCESS] Pipeline project successfully executed and calculated.")
