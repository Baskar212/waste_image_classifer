

---

```markdown
# 🗑️ Waste Image Classifier

This project focuses on classifying waste images into predefined categories using deep learning models.
It includes two approaches — a custom CNN and a pretrained ResNet-18 — and provides a user-friendly interface for real-time prediction and confidence display.

🔗 **GitHub Repository**: [waste_image_classifer](https://github.com/Baskar212/waste_image_classifer.git)  
📒 **Final Jupyter Notebook**: `FINAL_dash.ipynb`

---

## 🧠 Project Overview

This project involves:

- Classifying waste images using:
  - Custom CNN model
  - Transfer Learning with **ResNet-18**
- GUI-based image upload and prediction
- Exporting predictions and confidence scores to Excel

---


---

## 📊 Flow Diagrams

### 🔁 CNN Model Training  
cnn_model_training.png

---

### 🔁 ResNet18 Model Training  
resnet18_model_training.png

---

### 🚀 Final Project Flow Using ResNet18  
flow_diagram_using_resnet18_final.png

---

## ✅ Features

- 📦 Trained on the **TrashBox** dataset
- 🧠 Two models:
  - CNN from scratch
  - ResNet18 with fine-tuning
- 📊 Shows **prediction with confidence score**
- 📤 Allows exporting results to Excel
- 🖼️ Clears uploaded images and predictions (optional)

---

🛠️ Tools and Technologies Used
-PyTorch – Core deep learning framework used to build and train models
-TorchVision – Used for image transforms and pretrained models (ResNet18, ResNet50)
-dash (by Plotly) – For building the interactive web interface for predictions
-scikit-learn – Used for dataset splitting (train_test_split)
-Pandas – For handling tabular data
-PIL (Python Imaging Library) – For loading and processing images
-Plotly Express – For visualizations
-Google Colab – Used as the development and execution environment

🧠 Models Used
-ResNet-18 – Main model used for final implementation (pretrained, fine-tuned)
-ResNet-50 – Used for testing performance and comparisons
-Custom CNN – Built and tested for benchmarking against pretrained models

---

---

## 📌 Note

This project was developed for academic demonstration and may require further tuning for real-world deployment.

```

```
