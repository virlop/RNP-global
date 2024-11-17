import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import torch
from torchvision import models

def load_model(weights_path):
    """Carga un modelo FCN con los pesos especificados."""
    model = models.segmentation.fcn_resnet101(weights=None, num_classes=2, aux_loss=True)
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def preprocess_single_image(image):
    """Preprocesa una imagen para ser ingresada al modelo."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image)

def postprocess_output(output):
    """Convierte la salida del modelo en una máscara binaria."""
    output = output['out'] 
    output = torch.sigmoid(output)
    output = (output[:, 1, :, :] > 0.5).float().squeeze(0).squeeze(0) 
    return output

def overlay_mask_on_image(image, mask):
    """Superpone la máscara en la imagen original."""
    image_np = np.array(image.resize(mask.shape[::-1]))  # Redimensionar la imagen original
    mask_colored = np.zeros_like(image_np)
    mask_colored[:, :, 1] = mask * 255  # Usar el canal verde para la máscara
    return (0.6 * image_np + 0.4 * mask_colored).astype(np.uint8)