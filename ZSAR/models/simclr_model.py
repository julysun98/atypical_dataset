import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import Config

class SimCLRModel(nn.Module):
    def __init__(self, pretrained=True):
        super(SimCLRModel, self).__init__()
        
        # Use ResNet50 as backbone
        self.encoder = models.resnet50(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
        # Projection head as described in SimCLR paper
        self.projection = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, Config.FEATURE_DIM)
        )
        
    def forward(self, x):
        # Get features from encoder
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)
        
        # Project features
        projected = self.projection(features)
        
        return F.normalize(projected, dim=1)
    
    def get_features(self, x):
        # Get features without projection (for downstream tasks)
        features = self.encoder(x)
        return torch.flatten(features, start_dim=1)
