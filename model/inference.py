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
    Return a human-readable region label and distance from centre.

    ┌──────────┬──────────┬──────────┬──────────┬──────────┐
    │ upper-lt │  upper-l │  upper   │ upper-r  │ upper-rt │
    ├──────────┼──────────┼──────────┼──────────┼──────────┤
    │ slightly │ slightly │          │ slightly │ slightly │
    │   lt     │    l     │  centre  │    r     │    rt    │
    ├──────────┼──────────┼── centre ┼──────────┼──────────┤
    │ slightly │ slightly │          │ slightly │ slightly │
    │   lb     │    l     │          │    r     │    rb    │
    └──────────┴──────────┴──────────┴──────────┴──────────┘
    """
    # relative position in [0,1]
    rel_x, rel_y = x / W, y / H

    # fine grid thresholds (5×5)
    #   0.00 0.20 0.40 0.60 0.80 1.00
    idx_x = int(min(4, rel_x // .20))
    idx_y = int(min(4, rel_y // .20))

    # map indices to words
    hori = ['left', 'slightly left', 'centre', 'slightly right', 'right']
    vert = ['upper', 'slightly upper', 'centre', 'slightly lower', 'lower']

    # small dead-centre (5 % of each dimension)
    if abs(rel_x - .5) < .05 and abs(rel_y - .5) < .05:
        region = 'dead-centre'
    else:
        region = f'{vert[idx_y]} {hori[idx_x]}'.replace('centre centre', 'centre')

    # Euclidean distance to screen centre
    dist = math.hypot(x - W/2, y - H/2)
    return region.strip(), dist

def draw_arrow(img: Image.Image, ang_xy, save=None, show=False):
    w, h      = img.size
    cx, cy    = w / 2, h / 2
    scale     = min(w, h) / 4
    ex, ey    = cx + ang_xy[0] * scale, cy + ang_xy[1] * scale

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.axis('off')

    # ─── draw a single arrow from centre to end-point ──────────────────
    ax.annotate(
        '',                     # no text
        xy=(ex, ey),            # arrow tip
        xytext=(cx, cy),        # arrow tail
        arrowprops=dict(
            arrowstyle='->',    # normal arrow-head
            linewidth=3,
            color='red',
            shrinkA=0, shrinkB=0   # don’t shorten at either end
        )
    )

    ax.set_title(f'angle ({ang_xy[0]:+.2f}, {ang_xy[1]:+.2f})')

    if save:
        fig.savefig(save, bbox_inches='tight')
        print(f'✓ saved {save}')
    # if show:
    #     plt.show()
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