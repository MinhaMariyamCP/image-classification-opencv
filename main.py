import numpy as np
import cv2
from torchvision import datasets
from sklearn.model_selection import train_test_split

from src.preprocess import preprocess, enhance_image
from src.train import train_models
from src.evaluate import evaluate_model
from src.visualize import show_predictions

# =========================
# LOAD CIFAR-10 (NO TENSORFLOW)
# =========================
print("Downloading CIFAR-10 dataset...")

train_data = datasets.CIFAR10(root='./data', train=True, download=True)
test_data = datasets.CIFAR10(root='./data', train=False, download=True)

# Convert to numpy arrays
X_train_raw = np.array([img for img, _ in train_data])
y_train = np.array([label for _, label in train_data])

X_test_raw = np.array([img for img, _ in test_data])
y_test = np.array([label for _, label in test_data])

# Merge dataset
X_all = np.concatenate([X_train_raw, X_test_raw])
y_all = np.concatenate([y_train, y_test])

print("Dataset loaded successfully!")

# =========================
# PREPROCESSING
# =========================
print("Preprocessing images...")
X = preprocess(X_all)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# =========================
# TRAIN MODELS
# =========================
print("Training models...")
svm, knn = train_models(X_train, y_train)

# =========================
# EVALUATION
# =========================
print("Evaluating models...")
evaluate_model(svm, X_test, y_test, "SVM")
evaluate_model(knn, X_test, y_test, "KNN")

# =========================
# VISUALIZATION
# =========================
print("Generating predictions visualization...")
show_predictions(svm, X_test_raw[:2000], X_test[:2000], y_test[:2000])

# =========================
# IMAGE ENHANCEMENT DEMO
# =========================
print("Showing image enhancement...")
gray, enhanced, blurred = enhance_image(X_train_raw[0])
combined = np.hstack([gray, enhanced, blurred])

cv2.imshow("Original | Enhanced | Blurred", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done! Check outputs folder for results.")