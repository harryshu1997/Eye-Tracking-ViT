#!/usr/bin/env python3
"""
gaze_tools.py  –  export | predict | infer   (upright-image version)

export   : sample N faces from MPIIFaceGaze_normalizad and draw ground-truth arrow
predict  : run a ViT checkpoint on N dataset frames, draw predicted arrow
infer    : run the same checkpoint on one external RGB crop

Example
-------
# 1. Visualise 5 ground-truth samples
python gaze_tools.py export  --root datasets/MPIIFaceGaze_normalizad --n 5 --out gt_pngs

# 2. Predict on 5 random dataset frames
python gaze_tools.py predict --root /home/monsterharry/Documents/eye-tracking-vit/Eye-Tracking-ViT/datasets/MPIIFaceGaze_normalizad \
                             --checkpoint checkpoints/vit_gaze_best.pth \
                             --n 5 --out pred_pngs --offset 500

# 3. Infer on a selfie
python gaze_tools.py infer   --image selfie.png \
                             --checkpoint checkpoints/vit_gaze_best.pth \
                             --width 1440 --height 900
"""
# ---------------------------------------------------------------------------#
import os, sys, argparse, math, glob, numpy as np, h5py, torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms
import random


try:
    from model import ViTGazePredictor
except ImportError:
    ViTGazePredictor = None   # allows export mode without model.py

# ╭───────────────────────────  helpers  ─────────────────────────────────╮ #
def draw_arrow(img: Image.Image, ang_xy: np.ndarray, colour='red'):
    """Draw a gaze vector on a PIL image in place; angles already upright."""
    draw = ImageDraw.Draw(img)
    cx, cy = img.width / 2, img.height / 2
    dx, dy = ang_xy          # upright: dx = x, dy = y
    length = min(img.width, img.height) / 4
    ex, ey = cx + dx * length, cy + dy * length
    draw.line((cx, cy, ex, ey), fill=colour, width=4)

    angle = math.atan2(ey - cy, ex - cx)
    ah = 18
    for s in (-1, 1):
        hx = ex - ah * math.cos(angle + s * math.pi / 6)
        hy = ey - ah * math.sin(angle + s * math.pi / 6)
        draw.line((ex, ey, hx, hy), fill=colour, width=4)
# ╰────────────────────────────────────────────────────────────────────────╯ #

# ---------------------------------------------------------------------------#
#  MODE 1  –  export ground-truth arrows
# ---------------------------------------------------------------------------#
def export_samples(root, out_dir, n=5):
    os.makedirs(out_dir, exist_ok=True)
    exported = 0
    for f in sorted(os.listdir(root)):
        if not (f.startswith("p") and f.endswith((".mat", ".h5"))): continue
        path = os.path.join(root, f)
        with h5py.File(path, "r") as hf:
            data = hf["Data"]["data"]; label = hf["Data"]["label"]
            s_axis = max(range(data.ndim), key=lambda i: data.shape[i])
            N      = data.shape[s_axis]          # total samples in this file
            idx    = random.randrange(N)         # 0 … N-1
            frame  = (data[idx] if s_axis == 0 else
                    data[..., idx] if s_axis == data.ndim - 1
                    else np.take(data, idx, axis=s_axis))
            print(f"   » picked sample {idx}/{N-1} from {frame}")

            frame  = np.squeeze(frame)
            if frame.shape[0]==3: frame = np.transpose(frame,(1,2,0))
            if frame.dtype!=np.uint8:
                frame = (frame*255 if frame.max()<=1 else frame).clip(0,255).astype(np.uint8)
            l_axis = max(range(label.ndim), key=lambda i:label.shape[i])
            gaze   = (label[0,:2] if l_axis==0 else label[:2,0]).astype(np.float32)

        img = Image.fromarray(frame)
        draw_arrow(img, gaze, colour='lime')
        out = os.path.join(out_dir, f"{os.path.splitext(f)[0]}_gt.png")
        img.save(out); print(f"✓ saved {out}")
        exported += 1
        if exported >= n: break
    if exported == 0: print("⚠ no samples exported")

# ---------------------------------------------------------------------------#
#  MODE 2  –  predict arrows with checkpoint
# ---------------------------------------------------------------------------#
def load_model(ckpt_path, device):
    if ViTGazePredictor is None:
        sys.exit("model.py not importable; predict/infer unavailable.")
    model = ViTGazePredictor('vit_base_patch16_224', pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict']
                          if 'model_state_dict' in state else state)
    model.eval(); return model

TF = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
])

def predict_samples(root, ckpt, out_dir, n=5, offset=0):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(ckpt, device)

    exported = 0
    for f in sorted(os.listdir(root)):
        if not (f.startswith("p") and f.endswith((".mat",".h5"))): continue
        # ------- load first frame -----------------------------------------
        with h5py.File(os.path.join(root, f), "r") as hf:
            d = hf["Data"]["data"]
            s_axis = max(range(d.ndim), key=lambda i: d.shape[i])
            idx = offset % d.shape[s_axis]          # wrap around if > max
            frame = (d[idx] if s_axis == 0 else
                     d[..., idx] if s_axis == d.ndim - 1 else
                     np.take(d, idx, axis=s_axis))
            print(f"   » picked sample {idx}/{d.shape[s_axis]-1} from {f}")
            frame = np.squeeze(frame)
            if frame.shape[0]==3: frame = np.transpose(frame,(1,2,0))
            if frame.dtype!=np.uint8:
                frame = (frame*255 if frame.max()<=1 else frame).clip(0,255).astype(np.uint8)
        img_pil = Image.fromarray(frame)

        # ------- predict ---------------------------------------------------
        inp = TF(img_pil).unsqueeze(0).to(device)
        with torch.no_grad(): ang = model(inp)[0].cpu().numpy()

        vis = img_pil.copy(); draw_arrow(vis, ang, colour='red')
        out = os.path.join(out_dir,f"{os.path.splitext(f)[0]}_pred.png")
        vis.save(out); print(f"✓ saved {out}")
        exported += 1
        if exported>=n: break
    if exported==0: print("⚠ no samples predicted")

# ---------------------------------------------------------------------------#
#  MODE 3  –  infer on an external image
# ---------------------------------------------------------------------------#
def infer(image_path, ckpt, W, H):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(ckpt, device)

    img = Image.open(image_path).convert('RGB')
    ang = model(TF(img).unsqueeze(0).to(device))[0].cpu().numpy()

    x_px = (ang[0]+1)*0.5*W; y_px = (ang[1]+1)*0.5*H
    region = ("centre of screen" if abs(x_px-W/2)<.1*W and abs(y_px-H/2)<.1*H
              else f"{'top' if y_px<H*.33 else 'bottom' if y_px>H*.67 else 'centre'} "
                   f"{'left' if x_px<W*.33 else 'right' if x_px>W*.67 else 'centre'}")

    vis = img.copy(); draw_arrow(vis, ang)
    out = os.path.splitext(image_path)[0] + "_pred.png"; vis.save(out)

    print(f"angles  : ({ang[0]:+.3f},{ang[1]:+.3f})")
    print(f"pixels  : ({x_px:.1f},{y_px:.1f})  on {W}×{H}  → {region}")
    print(f"✓ saved {out}")

# ---------------------------------------------------------------------------#
#  CLI
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=True)

    e = sub.add_parser('export');  e.add_argument('--root', required=True); e.add_argument('--out', default='gt_pngs'); e.add_argument('--n', type=int, default=5)
    p = sub.add_parser('predict'); p.add_argument('--root', required=True); p.add_argument('--checkpoint', required=True); p.add_argument('--out', default='pred_pngs'); p.add_argument('--n', type=int, default=5)
    p.add_argument('--offset', type=int, default=0,
                   help='sample index to take from each .mat (0‑2999)')
    i = sub.add_parser('infer');   i.add_argument('--image', required=True); i.add_argument('--checkpoint', required=True); i.add_argument('--width', type=float, default=1440); i.add_argument('--height', type=float, default=900)

    args = ap.parse_args()
    if args.mode == 'export':
        export_samples(args.root, args.out, args.n)
    elif args.mode == 'predict':
        predict_samples(args.root, args.checkpoint, args.out, args.n, args.offset)
    else:
        infer(args.image, args.checkpoint, args.width, args.height)