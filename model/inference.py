import torch
import sys
from PIL import Image
from torchvision import transforms

from model import ViTCoordinateRegressor

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # 1. Define transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Load image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # shape [1, 3, 224, 224]

    # 3. Instantiate model & load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTCoordinateRegressor(model_name='vit_tiny_patch16_224.augreg_in21k_ft_in1k', num_outputs=2)
    model.load_state_dict(torch.load("checkpoints/vit_coordinate_regressor.pth", map_location=device))
    model.to(device)
    model.eval()

    # 4. Predict
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)  # shape [1, 2]
        pred_x, pred_y = output[0].cpu().tolist()

    print(f"Predicted coordinate: ({pred_x:.2f}, {pred_y:.2f})")

# Usage:
# python inference.py path/to/image.jpg
if __name__ == "__main__":
    main()