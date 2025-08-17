import torch
import random
import numbers
import numpy as np
from torchvision import transforms

class RandomTemporalSubsample:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        
    def __call__(self, frames):
        """
        Args:
            frames (tensor): Video frames (T, C, H, W)
        """
        total_frames = frames.shape[0]
        
        if total_frames <= self.num_frames:
            return frames
            
        start_idx = random.randint(0, total_frames - self.num_frames)
        return frames[start_idx:start_idx + self.num_frames]

class SpatialRandomCrop:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
        
    def __call__(self, frames):
        """
        Args:
            frames (tensor): Video frames (T, C, H, W)
        """
        h, w = frames.shape[-2:]
        th, tw = self.size
        
        if w == tw and h == th:
            return frames
            
        i = random.randint(0, h - th) if h > th else 0
        j = random.randint(0, w - tw) if w > tw else 0
        
        return frames[..., i:i + th, j:j + tw]

def get_augmentation_transforms(frame_size=224, num_frames=16):
    """Get transform pipeline for video frames."""
    return transforms.Compose([
        RandomTemporalSubsample(num_frames),
        SpatialRandomCrop(frame_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
