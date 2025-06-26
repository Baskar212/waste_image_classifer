import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained model and modify the final layer
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)  # Assuming 7 categories

# Load model weights
try:
    model.load_state_dict(torch.load('trashbox_model.pth', map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Waste categories
categories = ["Cardboard", "E-Waste", "Glass", "Medical", "Metal", "Paper", "Plastic"]

# Data storage for dashboard
if "waste_data" not in st.session_state:
    st.session_state.waste_data = {category: 0 for category in categories}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# File uploader
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the uploaded file to the same folder as the program
        file_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_files.append(file_path)
        st.success(f"Saved file: {uploaded_file.name}")

def classify_images(file_paths):
    results = []
    for file_path in file_paths:
        image = Image.open(file_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)
        _, predicted = torch.max(output, 1)
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item() * 100
        results.append((image, file_path, categories[predicted], confidence))
    return results

def generate_dashboard():
    fig, ax = plt.subplots()
    waste_data = st.session_state.waste_data
    categories = list(waste_data.keys())
    values = list(waste_data.values())
    ax.bar(categories, values)
    ax.set_xlabel('Waste Categories')
    ax.set_ylabel('Count')
    ax.set_title('Waste Collection Dashboard')
    return fig

def get_recycling_instructions():
    instructions = """
    - Cardboard: Flatten and place in recycling bin.
    - E-Waste: Take to an e-waste recycling center.
    - Glass: Rinse and place in recycling bin.
    - Medical: Follow local guidelines for disposal.
    - Metal: Rinse and place in recycling bin.
    - Paper: Place in recycling bin.
    - Plastic: Rinse and place in recycling bin.
    """
    return instructions

def remove_image(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    st.session_state.uploaded_files.remove(file_path)

if st.session_state.uploaded_files:
    classification_results = classify_images(st.session_state.uploaded_files)

    # Display each image with its classification result in a row
    st.subheader("Classification Results")
    for image, file_path, predicted_class, confidence in classification_results:
        col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
        with col1:
            st.image(image, caption=os.path.basename(file_path), width=150)  # Small preview
        with col2:
            st.write(f"**Predicted Class:** {predicted_class}")
        with col3:
            st.write(f"**Confidence:** {confidence:.2f}%")
        with col4:
            if st.button("❌", key=file_path):
                remove_image(file_path)
                st.experimental_rerun()
    
    # Display waste collection dashboard
    st.subheader("Waste Collection Dashboard")
    st.pyplot(generate_dashboard())

# Clear all uploaded images
if st.button("Clear All"):
    for file_path in st.session_state.uploaded_files:
        if os.path.exists(file_path):
            os.remove(file_path)
    st.session_state.uploaded_files = []
    st.success("All uploaded images have been removed.")
    st.experimental_rerun()

# Recycling instructions button
if st.button("Get Recycling Instructions"):
    st.subheader("Recycling Instructions")
    st.write(get_recycling_instructions())