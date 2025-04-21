import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import mediapipe as mp
import numpy as np
import cv2

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
def crop_eye_region_and_adjust_coords_raw(image_pil, gaze_x, gaze_y):
    width, height = image_pil.size
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    results = mp_face_mesh.process(image_cv)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        eye_indices = list(range(33, 134))  # both eyes

        eye_x = [landmarks[i].x * width for i in eye_indices]
        eye_y = [landmarks[i].y * height for i in eye_indices]

        margin = 20
        x_min = max(int(min(eye_x)) - margin, 0)
        y_min = max(int(min(eye_y)) - margin, 0)
        x_max = min(int(max(eye_x)) + margin, width)
        y_max = min(int(max(eye_y)) + margin, height)

        cropped_image = image_pil.crop((x_min, y_min, x_max, y_max))
        x_adj = gaze_x - x_min
        y_adj = gaze_y - y_min

        # Clamp (optional, just in case)
        x_adj = min(max(x_adj, 0), x_max - x_min)
        y_adj = min(max(y_adj, 0), y_max - y_min)

        return cropped_image, x_adj, y_adj

    # Fallback: return original
    return image_pil, gaze_x, gaze_y

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
        orig_x = entry["x"]
        orig_y = entry["y"]

        # Crop and adjust without normalization
        image, x_pixel, y_pixel = crop_eye_region_and_adjust_coords_raw(
            image, orig_x, orig_y
        )

        if self.training:
            jitter = 5.0  # jitter in pixels
            x_pixel += random.uniform(-jitter, jitter)
            y_pixel += random.uniform(-jitter, jitter)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor([x_pixel, y_pixel], dtype=torch.float32)
        return image, label