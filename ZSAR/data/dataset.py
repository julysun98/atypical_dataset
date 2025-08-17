import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import Config
from .video_transforms import get_augmentation_transforms
from .dataset_split import get_split_videos

class VideoDataset(Dataset):
    def __init__(self, split, transform=None):
        """
        Args:
            split (str): One of 'pretrain', 'train', or 'test'
            transform: Optional transform to be applied on a sample
        """
        self.split = split
        self.transform = transform if transform is not None else get_augmentation_transforms()
        self.videos = get_split_videos(split)
        
        # Create class to index mapping
        self.classes = sorted(list(set(class_name for _, class_name in self.videos)))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
    def _load_video(self, video_path):
        """Load video frames."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Uniformly sample NUM_FRAMES frames
        frame_indices = torch.linspace(0, total_frames - 1, Config.NUM_FRAMES).long()
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame to target size
                frame = cv2.resize(frame, (Config.FRAME_SIZE, Config.FRAME_SIZE))
                frames.append(frame)
            else:
                # If frame reading fails, append a blank frame
                frames.append(np.zeros((Config.FRAME_SIZE, Config.FRAME_SIZE, 3), dtype=np.uint8))
        
        cap.release()
        
        # Convert frames to torch tensor and change to (T, C, H, W) format
        frames = np.stack(frames)  # (T, H, W, C)
        frames = torch.from_numpy(frames)  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
        frames = frames.float() / 255.0  # Normalize to [0, 1]
        
        if self.transform:
            # Apply transforms to each frame independently
            transformed_frames = []
            for frame in frames:
                transformed_frames.append(self.transform(frame))
            frames = torch.stack(transformed_frames)
        
        return frames
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path, class_name = self.videos[idx]
        frames = self._load_video(video_path)
        class_idx = self.class_to_idx[class_name]
        
        # For pretraining, we create two augmented views of the same video
        if self.split == 'pretrain':
            frames2 = self._load_video(video_path)
            return {'frames': frames, 'frames2': frames2}
        else:
            return {
                'frames': frames,
                'class_idx': class_idx,
                'class_name': class_name  # ✅ 添加这一行
            }

def get_dataloader(split, batch_size=None, num_workers=None):
    """
    Get dataloader for specific split.
    
    Args:
        split (str): One of 'pretrain', 'train', or 'test'
        batch_size (int): Batch size (default: Config.BATCH_SIZE)
        num_workers (int): Number of workers (default: Config.NUM_WORKERS)
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS
    
    dataset = VideoDataset(split=split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split != 'test'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
