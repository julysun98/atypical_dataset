import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from datetime import datetime
from config import Config
from data.dataset import get_dataloader
from models.clip_model import CLIPVideoEncoder
from data.dataset_split import get_split_videos

def log_pretrain_sources():
    videos = get_split_videos('pretrain')

    base_dataset_name = Config.DATASET.lower()
    base_dataset_root = Config.UCF101_PATH if Config.DATASET == "UCF101" else Config.HMDB51_PATH

    count_base = sum(1 for v in videos if base_dataset_root in v[0])
    count_atypical = sum(1 for v in videos if 'atypical' in v[0].lower())
    count_k400 = sum(1 for v in videos if 'k400' in v[0].lower())

    total = len(videos)

    logging.info("[Pretraining Dataset Source Summary]")
    logging.info(f"  Total videos loaded: {total}")
    logging.info(f"  {Config.DATASET}:   {count_base}")
    logging.info(f"  Atypical: {count_atypical}")
    logging.info(f"  K400:     {count_k400}")


class VideoContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(VideoContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        batch_size = features.size(0) // 2

        similarity = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        similarity = similarity / self.temperature

        labels = torch.arange(batch_size, device=features.device)
        labels = torch.cat([
            labels + batch_size,
            labels
        ])

        mask = torch.eye(similarity.size(0), device=features.device)
        similarity = similarity.masked_fill(mask.bool(), float('-inf'))

        loss = F.cross_entropy(similarity, labels)
        return loss

def setup_logging():
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(Config.LOG_DIR, f'pretrain_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS}') as pbar:
        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(device, non_blocking=True)
            frames2 = batch['frames2'].to(device, non_blocking=True)

            frames = frames.to(dtype=next(model.parameters()).dtype)
            frames2 = frames2.to(dtype=next(model.parameters()).dtype)

            v1_features = model.encode_video(frames)
            v2_features = model.encode_video(frames2)

            features = torch.cat([v1_features, v2_features], dim=0)

            loss = VideoContrastiveLoss()(features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            current_loss = total_loss / (batch_idx + 1)

            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'batch': f'{batch_idx+1}/{num_batches}'
            })

    avg_loss = total_loss / num_batches
    return avg_loss

def main():
    setup_logging()
    logging.info(f"Starting pretraining with source: {Config.PRETRAIN_SOURCE}")

    log_pretrain_sources()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    train_loader = get_dataloader('pretrain')
    logging.info(f"Created dataloader with {len(train_loader.dataset)} samples")

    model = CLIPVideoEncoder(
        num_frames=Config.NUM_FRAMES,
        frame_size=Config.FRAME_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        num_heads=Config.NUM_HEADS,
        mlp_ratio=Config.MLP_RATIO,
        dropout=Config.DROPOUT
    ).to(device).float()
    logging.info("Initialized model")

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    logging.info("Starting training loop")
    for epoch in range(Config.NUM_EPOCHS):
        avg_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        logging.info(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % Config.SAVE_EVERY == 0:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'{Config.PRETRAIN_SOURCE}_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    main()
