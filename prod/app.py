import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter


st.title("Segmentación Semántica de Tumores Cerebrales")


# Cargar el modelo
@st.cache_resource
def get_model():
    model = load_model("prod/model.pth")
    model.eval()
    return model

model = get_model()

# Subir archivo
uploaded_file = st.file_uploader("Subí una imagen de una resonancia magnética cerebral y observá donde está el tumor 🧠 🔍 ", type=["png", "jpg", "jpeg"])


if uploaded_file:
    # Cargar y mostrar imagen original
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((300, 300))  # Cambia a tu tamaño deseado
    filtered_image = resized_image.filter(ImageFilter.EDGE_ENHANCE)
    
    # Preprocesar imagen
    image_tensor = preprocess_single_image(image).unsqueeze(0)
    
    # Realizar la predicción
    with st.spinner("Realizando segmentación..."):
        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = postprocess_output(output)
    
    # Visualizar la máscara predicha
    # Convertir la máscara predicha a NumPy y escalarla a 0-255
    pred_mask_np = (pred_mask.numpy() * 255).astype(np.uint8)
    pred_mask_img = Image.fromarray(pred_mask_np)
    pred_mask_img_resized = pred_mask_img.resize((300, 300))
    # Mostrar las imágenes lado a lado
    col1, col2 = st.columns(2)


    with col1:
        st.image(filtered_image, caption="Imagen Original", use_container_width=True)

    with col2:
        st.image(pred_mask_img_resized, caption="Predicción", use_container_width=True)

    # Ajustar la intensidad de la superposición
    st.title("¡Podés ajustar la intensidad de la superposición!")
    alpha = st.slider("Ajustá la intensidad del verde (correspondiente al tumor)", 0.0, 1.0, 0.4, 0.05)
    overlay_image = overlay_mask_on_image(resized_image, pred_mask_np, alpha=alpha)


    # Mostrar la imagen superpuesta
    st.image(overlay_image, caption=f"Imagen con Máscara Superpuesta (Intensidad: {alpha})", use_container_width=False)





            
