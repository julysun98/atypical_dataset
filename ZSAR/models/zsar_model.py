# models/zsar_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.text_encoder import TextEncoder
from models.clip_model import CLIPVideoEncoder
from config import Config

class ZSARModel(nn.Module):
    def __init__(self, device="cuda"):
        super(ZSARModel, self).__init__()
        self.video_encoder = CLIPVideoEncoder(
            num_frames=Config.NUM_FRAMES,
            frame_size=Config.FRAME_SIZE,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            num_heads=Config.NUM_HEADS,
            mlp_ratio=Config.MLP_RATIO,
            dropout=Config.DROPOUT,
            device=device
        ).to(device)

        self.text_encoder = TextEncoder(device=device)
        self.temperature = nn.Parameter(torch.tensor(0.07))  # learnable scaling factor
        self.device = device

    def build_prompts(self, class_names):
        return [f"a video of a person doing {name.replace('_', ' ')}" for name in class_names]

    def forward(self, videos, class_names):
        video_features = self.video_encoder.encode_video(videos)  # (B, D)
        prompts = self.build_prompts(class_names)
        text_features = self.text_encoder(prompts)                # (C, D)

        # Ensure consistent dtype
        video_features = F.normalize(video_features, dim=1).float()
        text_features = F.normalize(text_features, dim=1).float()

        similarity = torch.matmul(video_features, text_features.T) / self.temperature
        return similarity

    def compute_loss(self, similarity, labels):
        return F.cross_entropy(similarity, labels)
