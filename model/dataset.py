import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class JSONCoordinateDataset(Dataset):
    """
    Custom dataset that loads image and (x, y) coordinates from JSON.
    Applies label jittering if dataset.training = True.
    """
    def __init__(self, root_dir, transform=None, default_width=3024, default_height=1964):
        self.transform = transform
        self.entries = []
        self.training = False  # must be set externally

        for person in os.listdir(root_dir):
            person_path = os.path.join(root_dir, person)
            if not os.path.isdir(person_path):
                continue

            for session in os.listdir(person_path):
                session_path = os.path.join(person_path, session)
                if not os.path.isdir(session_path):
                    continue

                json_path = os.path.join(session_path, "capture_data.json")
                if os.path.isfile(json_path):
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    width = data.get("width", default_width)
                    height = data.get("height", default_height)
                    for item in data.get("image_data", []):
                        full_img_path = os.path.join(session_path, item["filename"])
                        entry = {
                            "image_path": full_img_path,
                            "x": float(item["x"]),
                            "y": float(item["y"]),
                            "width": float(width),
                            "height": float(height)
                        }
                        self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(entry["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        x_norm = entry["x"] / entry["width"]
        y_norm = entry["y"] / entry["height"]

        if self.training:
            jitter = 0.02
            x_norm += random.uniform(-jitter, jitter)
            y_norm += random.uniform(-jitter, jitter)
            x_norm = min(max(x_norm, 0.0), 1.0)
            y_norm = min(max(y_norm, 0.0), 1.0)

        label = torch.tensor([x_norm, y_norm], dtype=torch.float32)
        return image, label