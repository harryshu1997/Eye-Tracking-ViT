import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image

class CoordinateDataset(Dataset):
    """
    A custom PyTorch dataset that recursively scans a directory
    structure to find images named x_y.jpg, where x and y are coordinates.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the top-level data folder.
            transform (callable, optional): Optional transform to apply to each image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Regex to parse 'x_y.jpg'
        self.pattern = re.compile(r"(\d+)_(\d+)\.jpg", re.IGNORECASE)

        # Recursively scan all subfolders
        for person_folder in os.listdir(root_dir):
            person_path = os.path.join(root_dir, person_folder)
            if not os.path.isdir(person_path):
                continue

            for session_folder in os.listdir(person_path):
                session_path = os.path.join(person_path, session_folder)
                if not os.path.isdir(session_path):
                    continue

                for fname in os.listdir(session_path):
                    if not fname.lower().endswith('.jpg'):
                        continue

                    match = self.pattern.match(fname)
                    if match:
                        x_str, y_str = match.groups()
                        x_coord = float(x_str)
                        y_coord = float(y_str)

                        img_path = os.path.join(session_path, fname)
                        self.image_paths.append(img_path)
                        self.labels.append((x_coord, y_coord))
                    else:
                        print(f"Warning: {fname} did not match x_y.jpg pattern")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        x_coord, y_coord = self.labels[idx]
        # Normalize coordinates according to your resolution 3024 x 1964 (14 inch mac pro)
        x_norm = x_coord / 3024.0
        y_norm = y_coord / 1964.0
        label = torch.tensor([x_norm, y_norm], dtype=torch.float32)
        return image, label