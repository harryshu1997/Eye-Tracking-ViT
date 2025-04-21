import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import JSONCoordinateDataset
from model import ViTCoordinateRegressor

def main():
    # 1. Define paths
    data_dir = "../data"  # path to your data folder
    json_path = os.path.join(data_dir, "capture_data.json")  # path to JSON file


    # 2. Define transforms (resize + normalization)
    # If your images are large, resizing to 224x224 is typical for ViT.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # ImageNet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    ])

    # 3. Create dataset
    dataset = JSONCoordinateDataset(root_dir=data_dir, transform=transform)
    print(f"Total samples found: {len(dataset)}")

    # Optional: If your (x, y) range is large (e.g., up to 1920x1080),
    # consider normalizing them to [0,1] in the dataset __getitem__.
    # Then scale predictions back at inference time.

    # 4. Split dataset into train/val
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Enable label jittering for training only
    train_dataset.dataset.training = True
    val_dataset.dataset.training = False

    # 5. Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 6. Instantiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTCoordinateRegressor(model_name='timm/vit_base_patch16_224', num_outputs=2)
    model.to(device)

    # 7. Define loss & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 8. Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)  # shape [B, 2]

            optimizer.zero_grad()
            outputs = model(images)     # shape [B, 2]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")

    # 9. Save model
    os.makedirs("checkpoints", exist_ok=True)
    model_path = "checkpoints/vit_coordinate_regressor.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Usage:
# pip install torch torchvision timm
# python train.py
if __name__ == "__main__":
    main()