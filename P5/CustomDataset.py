import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['beaver', 'elephant', 'peppers', 'bear', 'dolphin', 'roses']
        self.class_idx = [1, 59, 25, 41,2, 13 ]

    def __len__(self):
        return len(self.classes) *10  # 10 images per class

    def __getitem__(self, idx):
        class_idx = idx // 10
        class_name = self.classes[class_idx]
        image_name = f"{class_name}{idx % 10 +1}.jpeg"
        image_path = os.path.join(self.root_dir, image_name)
        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            image_name = f"{class_name}{idx % 10 +1}.jpg"
            image_path = os.path.join(self.root_dir, image_name)
            image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, self.class_idx[class_idx]
