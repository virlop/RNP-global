import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import os

st.title("Segmentación Semántica de Tumores Cerebrales")
st.write("Carga una imagen de resonancia magnética (MRI) y obtén la segmentación del tumor.")

