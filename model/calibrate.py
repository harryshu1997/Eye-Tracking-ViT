#!/usr/bin/env python3
"""
calibrate.py
Fine-tune the last layer of ViTGazePredictor on your personal recordings.

Folder layout expected:

data/
└── person_1/
    ├── 1/
    │   ├── capture_data.json
    │   ├── image0.jpg
    │   └── ...
    └── 2/
        └── ...
└── person_2/
    └── ...

Each `capture_data.json` looks like

{
  "width": 1470,
  "height": 753,
  "image_data": [
    {"x": 180, "y": 120, "filename": "image0.jpg"},
    {"x": 1340, "y": 400, "filename": "image1.jpg"}
  ]
}
"""

# ---------------------------------------------------------------------------#
import os, json, glob, random, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision       import transforms
from PIL import Image
import torch.nn as nn, torch.optim as optim

from model import ViTGazePredictor          # your big net
# ---------------------------------------------------------------------------#

# ------------ edit these four lines ---------------------------------------
DATA_ROOT   = "/home/monsterharry/Documents/eye-tracking-vit/Eye-Tracking-ViT/data"         # folder that contains all persons
CHECKPOINT  = "checkpoints/vit_gaze_best.pth"   # model from main training
# ---------------------------------------------------------------------------#

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------------------------------------------------------------------#
#  convert old JSON format to new format
# ---------------------------------------------------------------------------#
for js in glob.glob(os.path.join(DATA_ROOT, "**", "capture_data.json"), recursive=True):
    with open(js) as f:
        data = json.load(f)
    # Skip if already in new format
    if "image_data" in data:
        continue
    # Convert old format
    image_data = []
    width = data.get("width", 1440)
    height = data.get("height", 900)
    for fname, v in data.items():
        if not fname.endswith(".jpg"):
            continue
        x = v.get("x_px") or v.get("x") or 0
        y = v.get("y_px") or v.get("y") or 0
        image_data.append({"x": x, "y": y, "filename": fname})
    new_data = {
        "width": width,
        "height": height,
        "image_data": image_data
    }
    with open(js, "w") as f:
        json.dump(new_data, f, indent=2)
    print(f"Converted {js}")

# ---------------------------------------------------------------------------#
#  dataset for calibration frames
# ---------------------------------------------------------------------------#
# ------------------------------------------------------------------
#  calibration dataset that understands the new JSON layout
# ------------------------------------------------------------------
class CalibDataset(Dataset):
    """
    Walk data/<person>/<session>/capture_data.json and return
    (image_tensor, gaze_angle) pairs.

    A JSON must contain keys:
        • width   : screen width   in pixels
        • height  : screen height  in pixels
        • image_data : list of {x, y, filename}
    """

    def __init__(self, root_dir: str, transform=None):
        self.items     = []                 # [(img_path, angle_xy)]
        self.transform = transform
        self._index(root_dir)

    # --------------------------------------------------------------
    def _index(self, root):
        json_files = glob.glob(os.path.join(root, "**", "capture_data.json"),
                               recursive=True)
        
        for js in json_files:
            
            with open(js) as f:
                meta = json.load(f)
            

            W = meta.get("width")
            H = meta.get("height")
            frames = meta.get("image_data", [])

            if W is None or H is None:
                print(f"⚠  {js} missing 'width' or 'height' – skipped")
                continue
            if not frames:
                print(f"⚠  {js} has no 'image_data' – skipped")
                continue

            sess_dir = os.path.dirname(js)
            for fr in frames:
                img_path = os.path.join(sess_dir, fr["filename"])
                if not os.path.isfile(img_path):
                    print(f"⚠  missing {img_path} – skipped"); continue

                # pixel (x,y) → angle (-1 … 1)
                x_ang = (fr["x"] / W) * 2 - 1
                y_ang = (fr["y"] / H) * 2 - 1
                self.items.append((img_path,
                                   np.array([x_ang, y_ang], dtype=np.float32)))

        print(f"Calibration set: {len(self.items)} frames "
              f"from {len(json_files)} session(s)")

    # --------------------------------------------------------------
    def __len__(self):  return len(self.items)

    def __getitem__(self, idx):
        path, ang = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img) if self.transform else transforms.ToTensor()(img)
        return img, torch.tensor(ang, dtype=torch.float32)

# ---------------------------------------------------------------------------#
#  transforms (same normalisation as training)
# ---------------------------------------------------------------------------#
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])

cal_ds = CalibDataset(root_dir=DATA_ROOT, transform=tf)
cal_loader = DataLoader(cal_ds, batch_size=8, shuffle=True)

# ---------------------------------------------------------------------------#
#  load model and freeze backbone
# ---------------------------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ViTGazePredictor("vit_base_patch16_224", pretrained=False).to(device)

try:
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state['model_state_dict']
                        if 'model_state_dict' in state else state)
    print("✓ base checkpoint loaded")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    print(f"Make sure {CHECKPOINT} exists and is compatible with your model")
    exit(1)

for p in model.parameters():          # freeze everything
    p.requires_grad_(False)

# Unfreeze the gaze_head instead of head (which doesn't exist in ViTGazePredictor)
for p in model.gaze_head.parameters():
    p.requires_grad_(True)
print("✓ model backbone frozen, gaze_head unfrozen for fine-tuning")

# ---------------------------------------------------------------------------#
#  fine-tune
# ---------------------------------------------------------------------------#
opt   = optim.AdamW(model.gaze_head.parameters(), lr=3e-4)
lossf = nn.MSELoss()

EPOCHS = 12
model.train()
for epoch in range(1, EPOCHS+1):
    running = 0.0
    for img, lbl in cal_loader:
        img, lbl = img.to(device), lbl.to(device)
        opt.zero_grad()
        loss = lossf(model(img), lbl)
        loss.backward(); opt.step()
        running += loss.item() * img.size(0)
    print(f"epoch {epoch:02d} | loss {running/len(cal_ds):.4f}")

# ---------------------------------------------------------------------------#
#  save
# ---------------------------------------------------------------------------#
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/vit_gaze_personal.pth")
print("✓ personalised model saved to checkpoints/vit_gaze_personal.pth")