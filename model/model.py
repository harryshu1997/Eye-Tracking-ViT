import timm
import torch
import torch.nn as nn

class ViTGazePredictor(nn.Module):
    """
    Vision Transformer model specifically for gaze direction prediction.
    Focuses only on predicting the 2D gaze direction vectors from the MPIFaceGaze dataset.
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(ViTGazePredictor, self).__init__()
        
        # Load pre-trained ViT model
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Get the output dimension of the backbone
        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()  # Remove the classification head
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the classification head
        else:
            raise ValueError(f"Model {model_name} structure not recognized")
        
        # Gaze direction prediction head
        self.gaze_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # 2D gaze direction vector
        )
    
    def forward(self, x):
        # Extract features through backbone
        features = self.backbone(x)
        
        # Predict gaze direction
        gaze = self.gaze_head(features)
        
        return gaze


# Alternative lighter implementation using the original structure
class ViTCoordinateRegressor(nn.Module):
    """
    A simple Vision Transformer model that outputs (x, y) gaze direction.
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(ViTCoordinateRegressor, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # Get the output dimension
        if hasattr(self.model, 'head'):
            in_features = self.model.head.in_features
        elif hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
        else:
            raise ValueError(f"Model {model_name} structure not recognized")
        
        # Improved head: MLP with normalization and dropout
        self.model.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 2D gaze direction
        )
    
    def forward(self, x):
        return self.model(x)