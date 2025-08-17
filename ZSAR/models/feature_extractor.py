import torch
import torch.nn as nn
import torchvision.models as models
from config import Config

class VideoFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(VideoFeatureExtractor, self).__init__()
        # Use ResNet50 as backbone
        base_model = models.resnet50(pretrained=pretrained)
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        
        # Add temporal modeling
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Project to desired feature dimension
        self.projection = nn.Sequential(
            nn.Linear(2048, Config.FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, 3, H, W)
        batch_size, num_frames, C, H, W = x.shape
        
        # Reshape for frame-wise processing
        x = x.view(batch_size * num_frames, C, H, W)
        
        # Extract frame-level features
        features = self.base_model(x)  # (batch_size * num_frames, 2048, 1, 1)
        
        # Reshape back to include temporal dimension
        features = features.view(batch_size, num_frames, -1)  # (batch_size, num_frames, 2048)
        
        # Temporal pooling
        features = features.transpose(1, 2).unsqueeze(-1)  # (batch_size, 2048, num_frames, 1)
        features = self.temporal_pool(features)  # (batch_size, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (batch_size, 2048)
        
        # Project to desired dimension
        features = self.projection(features)  # (batch_size, feature_dim)
        
        return features
