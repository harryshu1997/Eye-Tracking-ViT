import torch
import sys
import math
from PIL import Image
from torchvision import transforms
from model import ViTCoordinateRegressor

# Define 16 direction names when 0° is north and angles increase clockwise.
DIRECTION_NAMES = [
    "正北",    # 0° to 22.5°
    "北偏东",  # 22.5° to 45°
    "东北",    # 45° to 67.5°
    "东偏北",  # 67.5° to 90°
    "正东",    # 90° to 112.5°
    "东偏南",  # 112.5° to 135°
    "东南",    # 135° to 157.5°
    "南偏东",  # 157.5° to 180°
    "正南",    # 180° to 202.5°
    "南偏西",  # 202.5° to 225°
    "西南",    # 225° to 247.5°
    "西偏南",  # 247.5° to 270°
    "正西",    # 270° to 292.5°
    "西偏北",  # 292.5° to 315°
    "西北",    # 315° to 337.5°
    "北偏西"   # 337.5° to 360°
]

def quantize_direction(angle_deg):
    """
    Quantizes an angle (in degrees, where 0° is north and increases clockwise)
    into one of 16 bins.
    """
    bin_width = 360 / 16  # 22.5° per bin
    bin_index = int(angle_deg // bin_width) % 16
    return DIRECTION_NAMES[bin_index]

def main():
    if len(sys.argv) < 4:
        print("Usage: python inference.py <image_path> <width> <height>")
        sys.exit(1)

    image_path = sys.argv[1]
    normal_width = float(sys.argv[2])
    normal_height = float(sys.argv[3])

    # 1. Define transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Load image and apply transforms
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 224, 224]

    # 3. Instantiate model & load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTCoordinateRegressor(model_name='timm/vit_base_patch16_224', num_outputs=2)
    model.load_state_dict(torch.load("checkpoints/vit_coordinate_regressor.pth", map_location=device))
    model.to(device)
    model.eval()

    # 4. Predict (model outputs normalized coordinates)
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)  # shape: [1, 2]
        pred_x_norm, pred_y_norm = output[0].cpu().tolist()

    # 5. Convert normalized predictions to pixel coordinates
    pred_x = pred_x_norm * normal_width
    pred_y = pred_y_norm * normal_height

    # 6. Compute direction relative to screen center.
    center_x = normal_width / 2.0
    center_y = normal_height / 2.0
    # Compute vector from center to prediction.
    dx = pred_x - center_x
    dy = pred_y - center_y
    distance = math.sqrt(dx**2 + dy**2)

    # Define threshold for "Rest" (if within threshold pixels of center, output "Rest")
    threshold = 100.0

    if distance < threshold:
        direction = "静止"  # Rest
    else:
        # In the image coordinate system, with (0,0) at top-left and y increasing downward,
        # to get a compass direction with 0° as north we compute:
        # angle = (atan2(-dx, -dy) + 360) % 360
        angle_rad = math.atan2(-dx, -dy)
        angle_deg = (math.degrees(angle_rad) + 360) % 360
        direction = quantize_direction(angle_deg)

    print(f"Predicted coordinate: ({pred_x:.2f}, {pred_y:.2f})")
    if direction == "静止":
        print(f"Direction: {direction} (within {threshold} pixels of center)")
    else:
        print(f"Direction: {direction}")

def predict(img : Image, width, height):

    # 1. Define transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Load image and apply transforms
    img_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 224, 224]

    # 3. Instantiate model & load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTCoordinateRegressor(model_name='timm/vit_base_patch16_224', num_outputs=2)
    model.load_state_dict(torch.load("checkpoints/vit_coordinate_regressor.pth", map_location=device))
    model.to(device)
    model.eval()

    # 4. Predict (model outputs normalized coordinates)
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)  # shape: [1, 2]
        pred_x_norm, pred_y_norm = output[0].cpu().tolist()

    # 5. Convert normalized predictions to pixel coordinates
    pred_x = pred_x_norm * width
    pred_y = pred_y_norm * height

    # 6. Compute direction relative to screen center.
    center_x = width / 2.0
    center_y = width / 2.0
    # Compute vector from center to prediction.
    dx = pred_x - center_x
    dy = pred_y - center_y
    distance = math.sqrt(dx**2 + dy**2)

    # Define threshold for "Rest" (if within threshold pixels of center, output "Rest")
    threshold = 100.0

    if distance < threshold:
        direction = "静止"  # Rest
    else:
        # In the image coordinate system, with (0,0) at top-left and y increasing downward,
        # to get a compass direction with 0° as north we compute:
        # angle = (atan2(-dx, -dy) + 360) % 360
        angle_rad = math.atan2(-dx, -dy)
        angle_deg = (math.degrees(angle_rad) + 360) % 360
        direction = quantize_direction(angle_deg)

    print(f"Predicted coordinate: ({pred_x:.2f}, {pred_y:.2f})")
    if direction == "静止":
        print(f"Direction: {direction} (within {threshold} pixels of center)")
    else:
        print(f"Direction: {direction}")
    return pred_x, pred_y, direction

# if __name__ == "__main__":
#     main()