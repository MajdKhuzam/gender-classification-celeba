import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from data_loader import load_data

# ── Configuration ─────────────────────────────────────────────────────────────

# DATA_BASE_DIR = os.path.join(__file__, "..", 'Data', 'gender-recognition-200k-images-celeba', 'Dataset')
DATA_BASE_DIR = '/home/majd/Desktop/Gender Classification CNN - CelebA/Data/gender-recognition-200k-images-celeba/Dataset'
# MODEL_SAVE_PATH = os.path.join(__file__, "..", 'Data', 'gender_classification_model.keras')
MODEL_SAVE_PATH = '/home/majd/Desktop/Gender Classification CNN - CelebA/Data'
IMG_WIDTH  = 128
IMG_HEIGHT = 128

# ── Load test data ────────────────────────────────────────────────────────────

test_path = os.path.join(DATA_BASE_DIR, 'Test')
test_images, test_labels = load_data(test_path, IMG_WIDTH, IMG_HEIGHT)
print(f"Loaded {len(test_images)} test samples.")

# ── Load model ────────────────────────────────────────────────────────────────

model = load_model(MODEL_SAVE_PATH)
print(f"Model loaded from: {MODEL_SAVE_PATH}")

# ── Evaluate ──────────────────────────────────────────────────────────────────

print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"\nFinal Test Loss:     {test_loss:.4f}")
print(f"Final Test Accuracy: {test_accuracy:.4f}")

# ── Detailed metrics ──────────────────────────────────────────────────────────

predictions = model.predict(test_images)
predicted_labels = (predictions > 0.5).astype(int).flatten()

print("\n--- Classification Report ---")
print(classification_report(
    test_labels, predicted_labels,
    target_names=['Female (0)', 'Male (1)'],
))

print("--- Confusion Matrix ---")
print(confusion_matrix(test_labels, predicted_labels))