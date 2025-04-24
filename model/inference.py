#!/usr/bin/env python3
# infer.py (upright version)
import os, sys, math, argparse, numpy as np, torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import ViTGazePredictor            # your ViT regressor
import math

# ---------------------------------------------------------------------------#
#  parameters that depend on your monitor
# ---------------------------------------------------------------------------#
DEF_WIDTH, DEF_HEIGHT = 1512, 950            # edit if needed

'''

python inference.py --show --auto-crop --checkpoint checkpoints/vit_gaze_personal.pth --image '/home/monsterharry/Documents/eye-tracking-vit/Eye-Tracking-ViT/data/zhihao_shu/4/image10.jpg' 
python inference.py --show --auto-crop --checkpoint checkpoints/vit_gaze_best.pth --image '/home/monsterharry/Documents/eye-tracking-vit/Eye-Tracking-ViT/data/zhihao_shu/4/image10.jpg' 

'''
# ---------------------------------------------------------------------------#
#  ─── utilities ──────────────────────────────────────────────────────────── #
# ---------------------------------------------------------------------------#
def angle_to_pixel(ang_xy, W, H):
    """(-1,1) gaze angles → screen-pixel coordinates."""
    x = (ang_xy[0] + 1.) * 0.5 * W
    y = (ang_xy[1] + 1.) * 0.5 * H
    return x, y

def classify_region(x, y, W, H):
    """
    Return:
      • code: str  –  '01' for top-left, incrementing left→right, top→bottom,
                     centre cell = '00'
      • dist: float – Euclidean distance from screen center
    Uses a 7×7 grid for higher sensitivity.
    """
    # 1) Relative in [0,1]
    rel_x, rel_y = x / W, y / H

    # 2) Grid size
    N = 7
    idx_x = int(min(N-1, rel_x * N))
    idx_y = int(min(N-1, rel_y * N))

    # 3) Centre cell check
    center = N // 2  # 3
    if idx_x == center and idx_y == center:
        code = '00'
    else:
        num = idx_y * N + idx_x + 1  # 1…49
        code = f'{num:02d}'

    # 4) Distance from centre
    dist = math.hypot(x - W/2, y - H/2)

    return code, dist

def draw_arrow(img: Image.Image, ang_xy, save=None, show=False):
    """
    Draw the gaze arrow and overlay all 5×5 region centers labeled with codes.
    """
    w, h = img.size
    cx, cy = w/2, h/2
    scale = min(w, h) / 4
    ex, ey = cx + ang_xy[0]*scale, cy + ang_xy[1]*scale

    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(img)
    ax.axis('off')

    # 1) plot all grid‐cell centers and codes
    for row in range(5):
        for col in range(5):
            # cell center
            gx = (col + 0.5) * w / 5
            gy = (row + 0.5) * h / 5

            # compute code: center cell = '00', else 01..25
            if row == 2 and col == 2:
                code = '00'
            else:
                num = row*5 + col + 1
                code = f'{num:02d}'

            # plot marker and label
            ax.plot(gx, gy, marker='.', color='white', markersize=8, alpha=0.6)
            ax.text(gx, gy, code, color='white', fontsize=6,
                    ha='center', va='center', alpha=0.8)

    # 2) draw a single red arrow
    ax.annotate('', xy=(ex,ey), xytext=(cx,cy),
                arrowprops=dict(arrowstyle='->', color='red', linewidth=3, shrinkA=0, shrinkB=0))

    ax.set_title(f'angle ({ang_xy[0]:+.2f}, {ang_xy[1]:+.2f})')

    if save:
        fig.savefig(save, bbox_inches='tight')
        print(f'✓ saved {save}')
    if show:
        plt.show()
    plt.close(fig)

def auto_crop_face(pil_img, margin=0.25):
    """Detect the largest face and centre-crop a square around it."""
    try:
        import cv2
    except ImportError:
        print("opencv-python not installed; auto-crop disabled.")
        return pil_img
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    det  = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = det.detectMultiScale(gray, 1.1, 4)
    if len(faces)==0:
        print("⚠ no face detected; using full frame.")
        return pil_img
    # pick the largest box
    x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
    # expand to square
    side = int(max(w,h)*(1+margin))
    cx,cy = x+w//2, y+h//2
    left  = max(0, cx-side//2); upper = max(0, cy-side//2)
    right = min(pil_img.width,  left+side)
    lower = min(pil_img.height, upper+side)
    return pil_img.crop((left,upper,right,lower))

# ---------------------------------------------------------------------------#
#  1.  CLI
# ---------------------------------------------------------------------------#
ap = argparse.ArgumentParser(description="Gaze inference (upright images)")
ap.add_argument('--image',      required=True)
ap.add_argument('--checkpoint', default='checkpoints/vit_gaze_best.pth')
ap.add_argument('--width',  type=float, default=DEF_WIDTH)
ap.add_argument('--height', type=float, default=DEF_HEIGHT)
ap.add_argument('--out',  default='gaze_vis.png')
ap.add_argument('--show', action='store_true')
ap.add_argument('--no-save', action='store_true')
ap.add_argument('--auto-crop', action='store_true',
                help='auto-detect & crop face before inference (needs opencv)')
args = ap.parse_args()

if not (os.path.isfile(args.image) and os.path.isfile(args.checkpoint)):
    sys.exit("❌ missing image or checkpoint")

# ---------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'▶ device: {device}')

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
])

model = ViTGazePredictor('vit_base_patch16_224', pretrained=False).to(device)
state = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(state['model_state_dict']
                      if 'model_state_dict' in state else state)
model.eval(); print("✓ model loaded")

# ---------------------------------------------------------------------------#
#  2.  preprocess & inference
# ---------------------------------------------------------------------------#
img = Image.open(args.image).convert('RGB')
if args.auto_crop: img = auto_crop_face(img)
img = img.transpose(Image.FLIP_LEFT_RIGHT)

inp = tf(img).unsqueeze(0).to(device)

with torch.no_grad():
    ang = model(inp)[0].cpu().numpy()         # (x,y)  in  [-1,1]
ang[1] = -ang[1]                        #  <-- add this line

scr_x, scr_y = angle_to_pixel(ang, args.width, args.height)
region, dist = classify_region(scr_x, scr_y, args.width, args.height)

print("\nraw model output:", ang)
print(f"pixel coords = ({scr_x:.1f}, {scr_y:.1f})  |  {region}  "
      f"|  dist {dist:.1f}px")

if not args.no_save or args.show:
    draw_arrow(img, ang,
               None if args.no_save else args.out,
               show=args.show)