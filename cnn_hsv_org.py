import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Parameters
DATASET_PATH = "Original"
IMG_SIZE = 64
CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
NUM_CLASSES = len(CLASS_NAMES)

# HSV Preprocessing (H+S channels only)
def load_images_hsv(dataset_path):
    data = []
    labels = []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(dataset_path, class_name)
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hs = hsv[:, :, :2]  # Keep only H and S channels
                hs = cv2.resize(hs, (IMG_SIZE, IMG_SIZE))
                hs = hs / 255.0
                data.append(hs)
                labels.append(label_idx)
    return np.array(data), np.array(labels)

# Load and preprocess images
X, y = load_images_hsv(DATASET_PATH)
y_cat = to_categorical(y, NUM_CLASSES)

# Adjust shape for model input (HxWx2 → HxWx3 by adding a zero channel)
X = np.concatenate([X, np.zeros((X.shape[0], IMG_SIZE, IMG_SIZE, 1))], axis=-1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

# Augmentation
aug = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.1
)

train_generator = aug.flow(X_train, y_train, batch_size=32, subset='training')
val_generator = aug.flow(X_train, y_train, batch_size=32, subset='validation')

# Transfer Learning with DenseNet201
base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(train_generator, validation_data=val_generator, epochs=15)

# Evaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {acc:.4f}\n")

# Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Binary Cancerous/Non-Cancerous Accuracy
y_true_binary = np.where(y_true == 0, 0, 1)
y_pred_binary = np.where(y_pred == 0, 0, 1)

non_cancer_acc = accuracy_score(y_true_binary == 0, y_pred_binary == 0)
cancer_acc = accuracy_score(y_true_binary == 1, y_pred_binary == 1)

print(f"Non-Cancerous Accuracy: {non_cancer_acc * 100:.2f}%")
print(f"Cancerous Accuracy: {cancer_acc * 100:.2f}%\n")

# Classification Report
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Accuracy Plot
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save model
model.save("leukemia_cnn_model_hsv(org).h5")
joblib.dump("leukemia_cnn_model_hsv(org).h5", "leukemia_cnn_model_hsv(org).pkl")
print("Model saved as 'leukemia_cnn_model_hsv(org).h5' and .pkl")
