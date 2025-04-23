# train.py  – upright-data version
import os, sys, gc, random, math, torch, numpy as np, matplotlib.pyplot as plt
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data     import DataLoader
from torchvision          import transforms

from dataset import MPIFaceGazeDataset, PersonalGazeDataset    # upright images
from model   import ViTGazePredictor      # ViT regressor
# ---------------------------------------------------------------------------

# ╭───────────────────────────────── CONFIG ─────────────────────────────────╮
DATA_DIR    = "/home/monsterharry/Documents/eye-tracking-vit/Eye-Tracking-ViT/datasets/MPIIFaceGaze_normalizad"
SAMPLES_PP  = None       # None = all 3000
BATCH_SIZE  = 4
NUM_EPOCHS  = 3
USE_PIXELS  = True     # leave False unless you really want pixel targets
SCREEN_W, SCREEN_H = 1440, 900
SEED        = 42
# ╰──────────────────────────────────────────────────────────────────────────╯

# ---------------------------------------------------------------------------#
#  helpers
# ---------------------------------------------------------------------------#
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed()

def angle2pixel(ang: torch.Tensor) -> torch.Tensor:
    """Map (-1,1) angles → pixel coords (simple linear)."""
    x = (ang[..., 0] + 1) * 0.5 * SCREEN_W
    y = (ang[..., 1] + 1) * 0.5 * SCREEN_H
    return torch.stack((x, y), -1)

def to_target(t):                       # used later in training loop
    return angle2pixel(t) if USE_PIXELS else t

# ---------------------------------------------------------------------------#
#  transforms
# ---------------------------------------------------------------------------#
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),          # safe now (upright data)
    transforms.ColorJitter(.2, .2, .2),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])
val_tf   = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])

# ---------------------------------------------------------------------------#
#  dataset & split
# ---------------------------------------------------------------------------#
if not os.path.isdir(DATA_DIR):
    sys.exit(f"❌  dataset dir not found: {DATA_DIR}")

master = MPIFaceGazeDataset(DATA_DIR, transform=None,
                            limit_per_participant=SAMPLES_PP,
                            seed=SEED, apply_matlab_fix=True)
if len(master) == 0:
    sys.exit("❌  No samples indexed.")

val_size = int(0.2 * len(master))
indices  = list(range(len(master))); random.shuffle(indices)

train_ds = MPIFaceGazeDataset(DATA_DIR, transform=train_tf,
                              limit_per_participant=SAMPLES_PP,
                              seed=SEED, apply_matlab_fix=True)
train_ds.samples = [master.samples[i] for i in indices[:-val_size]]
train_ds.training = True

val_ds   = MPIFaceGazeDataset(DATA_DIR, transform=val_tf,
                              limit_per_participant=SAMPLES_PP,
                              seed=SEED, apply_matlab_fix=True)
val_ds.samples   = [master.samples[i] for i in indices[-val_size:]]

del master; gc.collect()

print(f"Train / Val = {len(train_ds)} / {len(val_ds)}")

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=1)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=1)

# ---------------------------------------------------------------------------#
#  model, loss, optim
# ---------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = ViTGazePredictor('vit_base_patch16_224', pretrained=True).to(device)
loss_fn = nn.MSELoss()
opt     = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

# ---------------------------------------------------------------------------#
#  train loop
# ---------------------------------------------------------------------------#
train_losses, val_losses = [], []
best_val = float('inf')
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs",     exist_ok=True)
print(f"▶ training on {device} …")

for epoch in range(1, NUM_EPOCHS + 1):
    # ---- train ------------------------------------------------------
    model.train(); run, n = 0.0, 0
    for i, (img, lbl) in enumerate(train_loader, 1):
        img, lbl = img.to(device), to_target(lbl.to(device))
        opt.zero_grad()
        out  = to_target(model(img))
        loss = loss_fn(out, lbl)
        loss.backward(); opt.step()

        run += loss.item() * img.size(0); n += img.size(0)
        if i % max(1, len(train_loader)//10) == 0:
            print(f"  epoch {epoch}  [{i}/{len(train_loader)}]  "
                  f"loss {loss.item():.4f}")
        del img, lbl, out, loss
        if device.type == 'cuda': torch.cuda.empty_cache()
    train_losses.append(run / n)

    # ---- validate ---------------------------------------------------
    model.eval(); run, n = 0.0, 0
    with torch.no_grad():
        for img, lbl in val_loader:
            img, lbl = img.to(device), to_target(lbl.to(device))
            loss = loss_fn(to_target(model(img)), lbl)
            run += loss.item() * img.size(0); n += img.size(0)
            del img, lbl, loss
            if device.type == 'cuda': torch.cuda.empty_cache()
    val_loss = run / n; val_losses.append(val_loss); sched.step(val_loss)

    print(f"✓ epoch {epoch}: train {train_losses[-1]:.4f} | val {val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "checkpoints/vit_gaze_best.pth")
        print("   ↳ saved new best")

# ---------------------------------------------------------------------------#
#  plot
# ---------------------------------------------------------------------------#
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='train'); plt.plot(val_losses, label='val')
plt.xlabel('epoch'); plt.ylabel('MSE' + (' (px)' if USE_PIXELS else ' (angle)'))
plt.legend(); plt.tight_layout()
plt.savefig("outputs/loss_curve.png"); plt.close()
print("✓ finished; curve saved to outputs/loss_curve.png")