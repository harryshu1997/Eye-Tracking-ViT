import os
import torch
import sys
import math
import numpy as np
from PIL import Image
from torchvision import transforms
from model import ViTGazePredictor
import argparse

# Load the ViT model
model = ViTGazePredictor(model_name='vit_base_patch16_224', pretrained=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define image transforms (same as in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load the saved model weights
checkpoint = torch.load("checkpoints/vit_gaze_predictor_final.pth", map_location=device)

if 'model_state_dict' in checkpoint:
    # If saved with full checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    # If saved with just state_dict
    model.load_state_dict(checkpoint)
model.to(device)
model.eval()

#def predict_gaze(model, image, device, transform):
def predict_gaze(image, width, height):

    """
    Run inference on a single image
    
    Args:
        model: The trained ViT model
        image: PIL Image
        device: torch device (cuda or cpu)
        transform: torchvision transforms
        
    Returns:
        normalized x, y gaze coordinates
    """
    # Convert image to tensor and normalize
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)  # shape: [1, 2]
        pred_x, pred_y = output[0].cpu().numpy()
    screen_x = pred_x * width
    screen_y = pred_y * height
    direction_info = calculate_screen_direction(screen_x, screen_y, width, height)

    print("\nGaze Prediction Results:")
    print(f"Normalized coordinates: ({pred_x:.4f}, {pred_y:.4f})")
    print(f"Screen coordinates: ({screen_x:.1f}, {screen_y:.1f})")
    print(f"Direction: {direction_info['direction']}")
    print(f"Distance from center: {direction_info['distance_from_center']:.1f} pixels")
    # Create results dictionary
    results = {
        "normalized_x": float(pred_x),
        "normalized_y": float(pred_y),
        "screen_x": float(screen_x),
        "screen_y": float(screen_y),
        "direction": direction_info["direction"],
        "distance_from_center": float(direction_info["distance_from_center"])
    }
    
    return results

def calculate_screen_direction(x, y, screen_width, screen_height):
    """
    Calculate the gaze direction relative to the screen (e.g., "top left", "bottom right")
    
    Args:
        x, y: Predicted gaze coordinates
        screen_width, screen_height: Screen dimensions
        
    Returns:
        dict with screen-relative direction and distance from center
    """
    # Calculate direction relative to center
    center_x, center_y = screen_width / 2.0, screen_height / 2.0
    
    # Vector from center to prediction
    dx = x - center_x
    dy = y - center_y
    distance = math.sqrt(dx**2 + dy**2)
    threshold = 100.0  # Threshold for "center" classification
    
    # Determine vertical position (top/center/bottom)
    if abs(dy) < threshold:
        vertical = "center"
    elif dy < 0:
        vertical = "top"
    else:
        vertical = "bottom"
    
    # Determine horizontal position (left/center/right)
    if abs(dx) < threshold:
        horizontal = "center"
    elif dx < 0:
        horizontal = "left"
    else:
        horizontal = "right"
    
    # Special case for center
    if vertical == "center" and horizontal == "center":
        direction = "center of screen"
    else:
        direction = f"{vertical} {horizontal} of screen"
    
    # Calculate distance from bounds for off-screen coordinates
    off_screen = ""
    if x < 0:
        off_screen = f" ({abs(x):.0f} pixels left of screen edge)"
    elif x > screen_width:
        off_screen = f" ({abs(x - screen_width):.0f} pixels right of screen edge)" 
    elif y < 0:
        off_screen = f" ({abs(y):.0f} pixels above screen edge)"
    elif y > screen_height:
        off_screen = f" ({abs(y - screen_height):.0f} pixels below screen edge)"
    
    # Add off-screen indication if applicable
    if off_screen:
        direction += off_screen
    
    return {
        "direction": direction,
        "distance_from_center": distance
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Gaze Direction Prediction')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='checkpoints/vit_gaze_predictor_best.pth', 
                      help='Path to trained model')
    parser.add_argument('--width', type=float, default=1470, help='Screen width')
    parser.add_argument('--height', type=float, default=753, help='Screen height')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} not found")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define image transforms (same as in training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Load model
    try:
        # Initialize model
        model = ViTGazePredictor(model_name='vit_base_patch16_224', pretrained=False)
        
        # Load the saved model weights
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            # If saved with full checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If saved with just state_dict
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load and process image
    try:
        image = Image.open(args.image).convert("RGB")
        print(f"Image loaded: {args.image}, size: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Run inference
    try:
        # Get normalized predictions
        pred_x, pred_y = predict_gaze(model, image, device, transform)
        
        # Convert to screen coordinates
        screen_x = pred_x * args.width
        screen_y = pred_y * args.height
        
        # Get direction
        direction_info = calculate_screen_direction(screen_x, screen_y, args.width, args.height)
        
        # Print results
        print("\nGaze Prediction Results:")
        print(f"Normalized coordinates: ({pred_x:.4f}, {pred_y:.4f})")
        print(f"Screen coordinates: ({screen_x:.1f}, {screen_y:.1f})")
        print(f"Direction: {direction_info['direction']}")
        print(f"Distance from center: {direction_info['distance_from_center']:.1f} pixels")
        
        # Create results dictionary
        results = {
            "normalized_x": float(pred_x),
            "normalized_y": float(pred_y),
            "screen_x": float(screen_x),
            "screen_y": float(screen_y),
            "direction": direction_info["direction"],
            "distance_from_center": float(direction_info["distance_from_center"])
        }
        
        return results
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()