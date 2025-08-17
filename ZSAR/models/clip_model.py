import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from config import Config

class TemporalTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class CLIPVideoEncoder(nn.Module):
    def __init__(self, num_frames=32, frame_size=224, hidden_size=512, 
                 num_layers=6, num_heads=8, mlp_ratio=4, dropout=0.1, device='cuda'):
        super().__init__()
        
        # Load CLIP model
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.device = device
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Temporal transformer for modeling frame sequence
        self.temporal_transformer = TemporalTransformer(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        ).to(device)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        ).to(device)
        
    def encode_video(self, x):
        """
        Args:
            x: Video tensor of shape (B, T, C, H, W)
        Returns:
            Video features of shape (B, D)
        """
        B, T = x.shape[:2]
        
        # Reshape input to (B*T, C, H, W) for CLIP processing
        x = x.view(B * T, *x.shape[2:])
        x = x.to(self.device)
        
        # Get CLIP features
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(x)  # (B*T, D)
        
        clip_features = clip_features.float()  # 👈 加这一句
        # Reshape back to (B, T, D)
        clip_features = clip_features.view(B, T, -1)
        
        # Apply temporal transformer
        temporal_features = self.temporal_transformer(clip_features)  # (B, T, D)
        
        # Pool temporal dimension
        video_features = temporal_features.mean(dim=1)  # (B, D)
        
        # Project features
        video_features = self.projection(video_features)  # (B, D)
        
        # Normalize features
        video_features = F.normalize(video_features, p=2, dim=-1)
        
        return video_features
