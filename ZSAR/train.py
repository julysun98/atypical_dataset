import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config
from data.dataset import get_dataloader
from data.dataset_split import load_splits
from models.zsar_model import ZSARModel

def train_epoch(model, train_loader, optimizer, device, base_class_names):
    model.train()
    total_loss = 0

    class2idx = {cls: i for i, cls in enumerate(base_class_names)}

    with tqdm(train_loader, desc='Training') as pbar:
        for batch in pbar:
            frames = batch['frames'].to(device)
            class_names = batch['class_name']  # Class names for each batch sample
            labels = torch.tensor([class2idx[c] for c in class_names], device=device)

            # Compute similarity between video and text embeddings
            similarity = model(frames, base_class_names)

            loss = model.compute_loss(similarity, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(train_loader)

def validate(model, val_loader, device, base_class_names):
    model.eval()
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    class2idx = {cls: i for i, cls in enumerate(base_class_names)}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            frames = batch['frames'].to(device)
            class_names = batch['class_name']
            labels = torch.tensor([class2idx[c] for c in class_names], device=device)

            similarity = model(frames, base_class_names)
            loss = model.compute_loss(similarity, labels)
            total_loss += loss.item()

            _, top1 = similarity.topk(1, dim=1)
            _, top5 = similarity.topk(5, dim=1)

            correct_top1 += top1.eq(labels.view(-1, 1)).sum().item()
            correct_top5 += top5.eq(labels.view(-1, 1)).sum().item()
            total += labels.size(0)

    acc1 = 100. * correct_top1 / total
    acc5 = 100. * correct_top5 / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, acc1, acc5

def main():
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")

    # Load base class names for training
    splits = load_splits()
    base_class_names = sorted(splits['train']['known_classes'])
    test_class_names = sorted(splits['test']['unknown_classes'])

    train_loader = get_dataloader(split='train')
    test_loader = get_dataloader(split='test')

    print(f"Train size: {len(train_loader.dataset)}, Test size: {len(test_loader.dataset)}")

    model = ZSARModel(device=device).to(device)

    # Load pretrained video encoder weights
    if Config.LOAD_PRETRAINED:
        path = os.path.join(Config.CHECKPOINT_DIR, f'{Config.PRETRAIN_SOURCE}_epoch_20.pt')
        print(f"[INFO] Loading pretrained weights from {path}")
        ckpt = torch.load(path, map_location=device)
        pretrained_dict = ckpt["model_state_dict"]

        clip_model_state = {
            k.replace("clip_model.", ""): v
            for k, v in pretrained_dict.items()
            if k.startswith("clip_model.")
        }

        missing, unexpected = model.video_encoder.clip_model.load_state_dict(clip_model_state, strict=False)
        print(f"✅ Loaded {len(clip_model_state)} keys into clip_model")
        if missing:
            print(f"⚠️ Missing keys: {missing}")
        if unexpected:
            print(f"⚠️ Unexpected keys: {unexpected}")

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    best_test_acc = 0

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, device, base_class_names)
        test_loss, test_acc_top1, test_acc_top5 = validate(model, test_loader, device, test_class_names)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Top-1: {test_acc_top1:.2f}%, Top-5: {test_acc_top5:.2f}%")

        if test_acc_top1 > best_test_acc:
            best_test_acc = test_acc_top1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc_top1': test_acc_top1,
                'test_acc_top5': test_acc_top5,
            }, os.path.join(Config.CHECKPOINT_DIR, f'{Config.PRETRAIN_SOURCE}_best_model.pth'))
            print(f"✅ Saved best model (Top-1 Acc: {test_acc_top1:.2f}%)")

    torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, f'{Config.PRETRAIN_SOURCE}_final_model.pth'))
    print("✅ Saved final model.")

if __name__ == '__main__':
    main()
