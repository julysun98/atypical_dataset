# models/text_encoder.py
import torch
import torch.nn as nn
import clip

class TextEncoder(nn.Module):
    def __init__(self, model_name='ViT-B/32', device=None):
        super(TextEncoder, self).__init__()
        self.model, _ = clip.load(model_name, device=device or "cuda")
        self.device = device or "cuda"

    def forward(self, prompts):
        """
        Args:
            prompts: List[str], e.g., ["a video of a person doing archery", ...]
        Returns:
            text_features: Tensor of shape (num_classes, feature_dim)
        """
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():  # freeze text encoder
            text_features = self.model.encode_text(tokens)
        return text_features  # shape: (C, D)
