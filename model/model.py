import timm
import torch
import torch.nn as nn

class ViTCoordinateRegressor(nn.Module):
    """
    A Vision Transformer model that outputs (x, y) coordinates.
    """
    def __init__(self, model_name='timm/vit_base_patch16_224', num_outputs=2):
        super(ViTCoordinateRegressor, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        in_features = self.model.head.in_features

        # Replace the classification head with a linear layer for regression
        self.model.head = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        return self.model(x)