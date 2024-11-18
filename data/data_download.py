import kagglehub
# Descargo el dataset de KAGGLE
path = kagglehub.dataset_download("nikhilroxtomar/brain-tumor-segmentation")
print("Path to dataset files:", path)

# Los datos descargados de este dataset se procesan en la clase dataset BrainTumorDataset.
class BrainTumorDataset(Dataset):
    def __init__(self, images_path, masks_path, transform, images_size):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_files = os.listdir(images_path)
        self.mask_files = os.listdir(masks_path)
        self.transform = transform
        self.images_size = images_size

        # Cargar todas las imágenes y máscaras en memoria
        self.images = [self.load_image(os.path.join(images_path, file)) for file in self.image_files]
        self.masks = [self.load_mask(os.path.join(masks_path, file)) for file in self.mask_files]

        assert len(self.images) == len(self.masks), "Número de imágenes y máscaras no son iguales"

    def load_image(self, path):
        image = Image.open(path).convert("RGB")
        image = image.resize(self.images_size)
        return self.transform(image)

    def load_mask(self, path):
        mask = Image.open(path).convert("L")
        mask = mask.resize(self.images_size)
        mask = np.array(mask) # Array numpy
        # Aplico un umbral
        umbral = 200
        mask = (mask >= umbral).astype(np.uint8)  # 1 para valores >= umbral (TUMOR), 0 en otro caso (FONDO)
        mask = torch.tensor(mask, dtype=torch.long)  # Convierto a tensor de tipo Long
        mask.unsqueeze_(0)  # Agrego una dimensión de canal
        return mask

    def unique_values(self, i):
        print("Unique values", torch.unique(self.masks[i]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.masks[i]

    def display_images_masks(self, i):
      fig, axs = plt.subplots(1, 2, figsize=(10, 5))
      axs[0].imshow(self.images[i].permute(1, 2, 0))
      axs[0].set_title('Imagen')
      axs[1].imshow(self.masks[i].squeeze(), cmap='gray')
      axs[1].set_title('Máscara')
      plt.show()