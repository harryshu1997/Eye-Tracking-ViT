import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class JSONCoordinateDataset(Dataset):
    """
    A custom PyTorch dataset that recursively scans a directory structure to find
    subfolders containing a "capture_data.json" file. The JSON file should have the structure:
    
    {
      "width": <image_width>,
      "height": <image_height>,
      "image_data": [
        { "x": <x_coordinate>, "y": <y_coordinate>, "filename": "<image_filename>" },
        ...
      ]
    }
    
    The dataset returns images and normalized (x, y) labels.
    
    Expected folder structure:
    
    data/
    └── Your_Name/
        ├── 0001/
        │   ├── capture_data.json
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── 0002/
        │   ├── capture_data.json
        │   ├── image3.jpg
        │   └── ...
        └── 0003/
            ├── capture_data.json
            ├── image4.jpg
            └── ...
    """
    def __init__(self, root_dir, transform=None, default_width=3024, default_height=1964):
        """
        Args:
            root_dir (str): Path to the top-level data folder.
            transform (callable, optional): Transformations to apply to each image.
            default_width (int): Default image width for normalization if JSON does not provide one.
            default_height (int): Default image height for normalization if JSON does not provide one.
        """
        self.transform = transform
        self.entries = []

        # Iterate over each person's folder in root_dir
        for person in os.listdir(root_dir):
            person_path = os.path.join(root_dir, person)
            if not os.path.isdir(person_path):
                continue

            # Iterate over each session folder for that person
            for session in os.listdir(person_path):
                session_path = os.path.join(person_path, session)
                if not os.path.isdir(session_path):
                    continue

                json_path = os.path.join(session_path, "capture_data.json")
                if os.path.isfile(json_path):
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    # Use JSON width/height if available, otherwise fall back to defaults.
                    width = data.get("width", default_width)
                    height = data.get("height", default_height)
                    for item in data.get("image_data", []):
                        # Each item should contain x, y, and filename.
                        full_img_path = os.path.join(session_path, item["filename"])
                        entry = {
                            "image_path": full_img_path,
                            "x": float(item["x"]),
                            "y": float(item["y"]),
                            "width": float(width),
                            "height": float(height)
                        }
                        self.entries.append(entry)
                else:
                    print(f"Warning: No capture_data.json found in {session_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(entry["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Normalize the coordinates based on the image dimensions (from JSON or default)
        x_norm = entry["x"] / entry["width"]
        y_norm = entry["y"] / entry["height"]
        label = torch.tensor([x_norm, y_norm], dtype=torch.float32)
        return image, label