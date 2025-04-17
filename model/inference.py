import torch
import sys
import math
import json
import os
from PIL import Image
from torchvision import transforms
from model import ViTCoordinateRegressor
import time
from sklearn.linear_model import LinearRegression
import numpy as np

calib_model = None
# Instantiate model & load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTCoordinateRegressor(model_name='timm/vit_base_patch16_224', num_outputs=2)
model.load_state_dict(torch.load("checkpoints/vit_coordinate_regressor.pth", map_location=device))
model.to(device)
model.eval()

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])
# 16 compass directions in English
DIRECTION_NAMES = [
    "North", "North-Northeast", "Northeast", "East-Northeast",
    "East", "East-Southeast", "Southeast", "South-Southeast",
    "South", "South-Southwest", "Southwest", "West-Southwest",
    "West", "West-Northwest", "Northwest", "North-Northwest"
]

def quantize_direction(angle_deg):
    bin_width = 360 / 16
    bin_index = int(angle_deg // bin_width) % 16
    return DIRECTION_NAMES[bin_index]

# Optional smoothing state
# prev_x, prev_y = None, None
# def smooth_coords(x, y, alpha=0.2):
#     global prev_x, prev_y
#     if prev_x is None:
#         prev_x, prev_y = x, y
#     else:
#         prev_x = alpha * x + (1 - alpha) * prev_x
#         prev_y = alpha * y + (1 - alpha) * prev_y
#     return prev_x, prev_y

def run_inference(img: Image.Image, width: float, height: float, smooth=False, move_cursor=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)



def fit_calibration_model(calibration_data):

    """
    Fits a calibration regression model from predicted → true coordinates.

    Args:
        calibration_data (List[Dict]): Output from calibrate_sample with keys:
            'pred_x', 'pred_y', 'true_x', 'true_y'
    Returns:
        sklearn.linear_model.LinearRegression: 2D regression model
    """
    X = np.array([[d["pred_x"], d["pred_y"]] for d in calibration_data])
    Y = np.array([[d["true_x"], d["true_y"]] for d in calibration_data])
    model = LinearRegression()
    model.fit(X, Y)
    return model

def run_inference_with_calibration(img: Image.Image, width: float, height: float,
                             smooth=False, move_cursor=False):

    """
    Performs gaze prediction and adjusts using a fitted calibration model.

    Args:
        img: input image (PIL)
        width, height: screen resolution
        calibration_model: sklearn model from fit_calibration_model
        smooth: whether to apply smoothing
        move_cursor: move mouse (optional)
    Returns:
        dict: predicted/calibrated gaze coordinates, direction, etc.
    """

    raw = predict(img, width, height)
    pred_x = raw["pred_x"]
    pred_y = raw["pred_y"]

    # Apply calibration model
    if calib_model is None:
        raise ValueError("Calibration model has not been initialized. Call /calibrate first.")

    calibrated = calib_model.predict([[pred_x, pred_y]])[0]
    calib_x, calib_y = calibrated.tolist()

    # if move_cursor:
    #     import pyautogui
    #     screen_x = min(max(0, calib_x), pyautogui.size().width)
    #     screen_y = min(max(0, calib_y), pyautogui.size().height)
    #     pyautogui.moveTo(int(screen_x), int(screen_y))

    center_x, center_y = width / 2.0, height / 2.0
    dx = calib_x - center_x
    dy = calib_y - center_y
    distance = math.sqrt(dx**2 + dy**2)

    # Define threshold for "Rest" (if within threshold pixels of center, output "Rest")
    threshold = 100.0
    if distance < threshold:
        direction = "Rest"
    else:
        angle_rad = math.atan2(-dx, -dy)
        angle_deg = (math.degrees(angle_rad) + 360) % 360
        direction = quantize_direction(angle_deg)
    print(f"Predicted coordinate: ({pred_x:.2f}, {pred_y:.2f})")
    print(f"[CALIBRATED] Coordinate: ({calib_x:.2f}, {calib_y:.2f}), Direction: {direction}")
    return {
        "pred_x": calib_x,
        "pred_y": calib_y,
        "direction": direction,
        "distance_from_center": distance
    }

def predict(img : Image, width, height):

    # 2. Load image and apply transforms
    img_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 224, 224]

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
    threshold = 100.0

    if distance < threshold:
        direction = "Rest"
    else:
        angle_rad = math.atan2(-dx, -dy)
        angle_deg = (math.degrees(angle_rad) + 360) % 360
        direction = quantize_direction(angle_deg)

    #print(f"[CALIBRATED] Coordinate: ({calib_x:.2f}, {calib_y:.2f}), Direction: {direction}")
    return {
        "pred_x": pred_x,
        "pred_y": pred_y,
        "direction": direction,
        "distance_from_center": distance
    }

def calibrate_sample(img: Image.Image, width: float, height: float,
                     true_x: float, true_y: float, smooth=False) -> dict:
    """
    Takes a calibration image and known screen location, returns model prediction + error.
    
    Args:
        img: PIL image of user's face.
        width, height: screen resolution in pixels.
        true_x, true_y: the actual gaze target location (in pixels).
        smooth: whether to apply smoothing (optional).
        
    Returns:
        dict with model prediction, true values, and error metrics.
    """
    result = predict(img, width, height)
    
    pred_x = result["pred_x"]
    pred_y = result["pred_y"]

    # Calculate Euclidean error in pixels
    error = math.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)

    return {
        "timestamp": time.time(),
        "true_x": true_x,
        "true_y": true_y,
        "pred_x": pred_x,
        "pred_y": pred_y,
        "direction": result["direction"],
        "error": error
    }

# Assuming image is from webcam and user looked at (960, 540)
# result = calibrate_sample(pil_image, width=1920, height=1080, true_x=960, true_y=540)
# print(result)

# Sample output:
# {
#     'timestamp': 1713198235.18898,
#     'true_x': 960,
#     'true_y': 540,
#     'pred_x': 1003.2,
#     'pred_y': 511.7,
#     'direction': 'East',
#     'error': 47.6
# }

'''
# TODO: @Khoa please follow this, it would be more accurate
Example of how to use calib function:
1. on the web UI, collect some pictures along with predicted (x,y) and ground truth (x1,y1)
samples = []
for img_path, (true_x, true_y) in zip(image_paths, true_coords):
    img = Image.open(img_path).convert("RGB")
    sample = calibrate_sample(img, 1920, 1080, true_x, true_y)
    samples.append(sample)

2. fit the calibration model
calib_model = fit_calibration_model(samples)

3. use the updated inference with calibration in the web UI
output = run_inference_with_calibration(img, 1920, 1080, calib_model, smooth=True)

'''


def initialize_calibration_model():

    """
    Initializes the calibration model to None.
    """
    
    samples = []
    folder_path = "temp_data"
    json_path = os.path.join(folder_path, "capture_data.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    screen_width = data["width"]
    screen_height = data["height"]
    image_paths = [os.path.join(folder_path, item["filename"]) for item in data["image_data"]]
    true_coords = [(item["x"], item["y"]) for item in data["image_data"]]
    for img_path, (true_x, true_y) in zip(image_paths, true_coords):
        img = Image.open(img_path).convert("RGB")
        sample = calibrate_sample(img, screen_width, screen_height, true_x, true_y)
        samples.append(sample)
    global calib_model 
    calib_model = fit_calibration_model(samples)
    return calib_model

# def main():
#     if len(sys.argv) < 4:
#         print("Usage: python inference.py <image_path> <width> <height>")
#         sys.exit(1)

#     image_path = sys.argv[1]
#     normal_width = float(sys.argv[2])
#     normal_height = float(sys.argv[3])

#     img = Image.open(image_path).convert("RGB")
#     run_inference(img, normal_width, normal_height, smooth=False, move_cursor=False)

# if __name__ == "__main__":
#     main()