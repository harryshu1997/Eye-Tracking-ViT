import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import gc  # Garbage collector

# Import the memory-efficient dataset and model
from dataset import MPIFaceGazeDataset
from model import ViTGazePredictor

def main():
    # 1. Define paths - use the exact path from error message
    data_dir = "/home/monsterharry/Documents/eye-tracking-vit/Eye-Tracking-ViT/datasets/MPIIFaceGaze_normalizad"
    
    if not os.path.exists(data_dir):
        print(f"Warning: Dataset directory {data_dir} does not exist!")
        alternate_dirs = [
            "datasets/MPIIFaceGaze_normalizad",
            "../datasets/MPIIFaceGaze_normalizad"
        ]
        
        for alt_dir in alternate_dirs:
            if os.path.exists(alt_dir):
                data_dir = alt_dir
                print(f"Using alternative path: {data_dir}")
                break
    
    # 2. Define transforms for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Create dataset with memory-efficient loading
    # Limit samples per participant for lower memory usage
    print(f"Loading dataset from {data_dir}...")
    
    try:
        # Limit samples to 100 per participant for test run
        samples_per_participant = 1000
        
        # Create dataset
        dataset = MPIFaceGazeDataset(
            root_dir=data_dir, 
            transform=None,  # No transform yet - will apply later
            use_all_labels=False,
            limit_per_participant=samples_per_participant
        )
        
        # Check if dataset loaded successfully
        if len(dataset) == 0:
            raise ValueError(f"No samples indexed from {data_dir}")
        
        # 4. Split dataset
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        
        # Generate indices for train/val split
        indices = list(range(len(dataset)))
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create datasets with appropriate transforms
        train_dataset = MPIFaceGazeDataset(
            root_dir=data_dir, 
            transform=train_transform,
            use_all_labels=False,
            limit_per_participant=samples_per_participant
        )
        train_dataset.samples = [dataset.samples[i] for i in train_indices]
        train_dataset.training = True  # Enable jittering
        
        val_dataset = MPIFaceGazeDataset(
            root_dir=data_dir, 
            transform=val_transform,
            use_all_labels=False,
            limit_per_participant=samples_per_participant
        )
        val_dataset.samples = [dataset.samples[i] for i in val_indices]
        val_dataset.training = False  # Disable jittering
        
        # Free up memory from the original dataset
        del dataset
        gc.collect()
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # 5. Create dataloaders with small batch size
        train_loader = DataLoader(
            train_dataset, 
            batch_size=4,  # Very small batch size to reduce memory
            shuffle=True,
            num_workers=1,  # Reduce workers to save memory
            pin_memory=False  # Disable pin_memory to save GPU memory
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=4,
            shuffle=False,
            num_workers=1,
            pin_memory=False
        )
        
        # 6. Instantiate model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Clear GPU memory before loading model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Use the gaze direction predictor model
        model = ViTGazePredictor(model_name='vit_base_patch16_224', pretrained=True)
        model.to(device)
        
        # 7. Define loss & optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # 8. Training loop
        num_epochs = 10  # Fewer epochs for testing
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        # Create output directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        
        start_time = datetime.now()
        print(f"Training started at {start_time}")
        
        for epoch in range(num_epochs):
            # ---- Training ----
            model.train()
            running_loss = 0.0
            samples_processed = 0
            
            for i, (images, labels) in enumerate(train_loader):
                try:
                    # Move data to device
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Track loss
                    batch_size = images.size(0)
                    running_loss += loss.item() * batch_size
                    samples_processed += batch_size
                    
                    # Print progress
                    if i % 5 == 0:
                        print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(train_loader)}, "
                              f"Loss: {loss.item():.6f}")
                    
                    # Explicitly clear memory
                    del images, labels, outputs, loss
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error in training batch {i}: {e}")
                    # Try to recover by clearing memory
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
            
            # Compute average loss
            if samples_processed > 0:
                train_loss = running_loss / samples_processed
                train_losses.append(train_loss)
            else:
                print("Warning: No samples processed in training!")
                train_losses.append(float('inf'))
            
            # ---- Validation ----
            model.eval()
            val_running_loss = 0.0
            val_samples_processed = 0
            
            with torch.no_grad():
                for i, (images, labels) in enumerate(val_loader):
                    try:
                        # Move data to device
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        # Forward pass
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                        # Track loss
                        batch_size = images.size(0)
                        val_running_loss += loss.item() * batch_size
                        val_samples_processed += batch_size
                        
                        # Explicitly clear memory
                        del images, labels, outputs, loss
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"Error in validation batch {i}: {e}")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
            
            # Compute average validation loss
            if val_samples_processed > 0:
                val_loss = val_running_loss / val_samples_processed
                val_losses.append(val_loss)
            else:
                print("Warning: No samples processed in validation!")
                val_losses.append(float('inf'))
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.8f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join("checkpoints", "vit_gaze_predictor_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }, model_path)
                print(f"New best model saved to {model_path}")
            
            # Force garbage collection between epochs
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Training complete
        end_time = datetime.now()
        training_time = end_time - start_time
        print(f"Training completed in {training_time}")
        
        # 9. Save final model
        final_model_path = os.path.join("checkpoints", "vit_gaze_predictor_final.pth")
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'best_val_loss': best_val_loss
        }, final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # 10. Plot training curves
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join("outputs", "training_loss_curve.png"))
        print("Training loss curve saved to outputs/training_loss_curve.png")
        
    except Exception as e:
        import traceback
        print(f"Error during execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()