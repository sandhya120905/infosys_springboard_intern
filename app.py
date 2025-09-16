import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Chest X-ray Classification")
st.write("Upload a Chest X-ray image to classify it.")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    input_tensor = transform(image).unsqueeze(0)
    model = load_model()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class_idx = torch.argmax(probs).item()
    class_names = ["Normal", "Pneumonia"]
    st.write(f"Predicted Class: {class_names[pred_class_idx]}")
    st.write(f"Confidence: {probs[pred_class_idx].item()*100:.2f}%")
    df = pd.DataFrame({
        "Class": class_names,
        "Probability": [prob.item() for prob in probs]
    }).set_index("Class")
    st.bar_chart(df)
