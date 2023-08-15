from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
import numpy as np

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    """
    WARNING: Do not perform data normalization here. 
    """
    def __init__(self, dataset_path):
        self.path = dataset_path

        # Images
        image_files = pd.read_csv(os.path.join(dataset_path, 'labels.csv'))['file'].tolist()
        self.image_paths = [os.path.join(dataset_path, x) for x in image_files]

        # Labels
        self.str_labels = pd.read_csv(os.path.join(dataset_path, 'labels.csv'))['label'].tolist()
        label_dict = {"background": 0, "kart": 1, "pickup": 2, "nitro": 3, "bomb": 4, "projectile": 5}
        self.int_labels = [label_dict[x] for x in self.str_labels]

    def __len__(self):
        return len(self.int_labels)

    def __getitem__(self, idx):
        label = self.int_labels[idx]

        # Convert filepath to tensor
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image)

        return image_tensor, label


def load_data(dataset_path, num_workers=4, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
