# Acute_Lymphoblastic_Leukemia_Detection_from_Blood_Smear_images_using_Deep_Learning

This repository presents a deep learning pipeline for the detection of Acute Lymphoblastic Leukemia (ALL) using peripheral blood smear (PBS) images. It uses the DenseNet201 architecture with transfer learning to achieve high accuracy in classifying leukemia vs healthy blood cells.

---

## 📌 Overview

Acute Lymphoblastic Leukemia is a severe form of blood cancer. Manual diagnosis using blood smear microscopy is time-consuming and prone to errors. This project introduces an automated solution using DenseNet201, a powerful convolutional neural network architecture pretrained on ImageNet, fine-tuned for this medical task.
- **Dataset** : [Kaggle](https://www.kaggle.com/datasets/mehradaria/leukemia)

---

## 🧪 Dataset Summary

- **Total Images:** 3,562 peripheral blood smear (PBS) images  
- **Patients:** 89 total  
  - 64 diagnosed with ALL  
  - 25 healthy (hematogone)  
- **Subtypes:** ALL-L1, ALL-L2, Hematogone  
- **Image Size:** All resized to 224×224×3  
- **Source:** Public ALL-IDB dataset  
- **Classes:**  
  - `leukemia/` – containing ALL images  
  - `healthy/` – containing benign images

---

## 🔧 Preprocessing

- HSV color space thresholding  
- ROI segmentation (nucleus-focused)  
- Rescaling to 224×224  
- Image normalization  
- Label encoding for binary classification

---

## 🧠 Model: DenseNet201 Architecture

- **Backbone:** DenseNet201 (include_top=False, pretrained on ImageNet)  
- **Custom Layers:**
  - GlobalAveragePooling2D  
  - Dense(256, activation='relu')  
  - Dropout(0.5)  
  - Dense(1, activation='sigmoid')  
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Training Strategy:**
  1. Freeze base layers
  2. Train top layers
  3. Unfreeze and fine-tune selected DenseNet201 layers

---

## 📊 Model Performance

| Metric               | Value   |
|----------------------|---------|
| Accuracy             | 91.26%  |

