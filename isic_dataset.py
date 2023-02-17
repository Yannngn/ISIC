import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class ISICDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_file = self.img_labels.iloc[idx, 0] + '.jpg'
        img_path = os.path.join(self.img_dir, img_file)
        
        image = read_image(img_path)
        
        label = self.img_labels.iloc[idx, 1]
        label = torch.tensor(label)
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label