
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import os
from utils.util import AverageMeter, accuracy
from tqdm import tqdm
from models.resnet3d import generate_model
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from data.ucfdataloader import ucf101_pace_pretrain
from data.video_transforms import RandomCrop, RandomHorizontalFlip, ClipResize
from torch.utils.tensorboard import SummaryWriter

def get_dynamic_sampler(dataset_size, sample_count):
    indices = torch.randperm(dataset_size)[:sample_count]
    return SubsetRandomSampler(indices)

def train(epoch, model, device, dataloader, optimizer, exp_lr_scheduler, criterion, args, writer):
    loss_record = AverageMeter()
    acc_record = AverageMeter()
    exp_lr_scheduler.step()
    model.train()
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        data, label = batch['frames'].to(device, dtype=torch.float), batch['label'].to(device)
        optimizer.zero_grad()
        _, output = model(data)
        loss = criterion(output, label)
        acc = accuracy(output, label)
        acc_record.update(acc[0].item(), data.size(0))
        loss_record.update(loss.item(), data.size(0))
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)
        writer.add_scalar('Accuracy/train', acc[0].item(), epoch * len(dataloader) + batch_idx)
    print('Train Epoch: {} Avg Loss: {:.4f} 	 Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))
    return loss_record

def test(model, device, dataloader, args):
    acc_record = AverageMeter()
    model.eval()
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        data, label = batch['frames'].to(device, dtype=torch.float), batch['label'].to(device)
        _, output = model(data)
        acc = accuracy(output, label)
        acc_record.update(acc[0].item(), data.size(0))
    print('Test Acc: {:.4f}'.format(acc_record.avg))
    return acc_record

def main():
    parser = argparse.ArgumentParser(description='UCF101 + Atypical Dynamic Sampling Training')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--video_dir_ucf101', type=str, required=True)
    parser.add_argument('--train_label_ucf101', type=str, required=True)
    parser.add_argument('--test_label_ucf101', type=str, required=True)
    parser.add_argument('--video_dir_atypical', type=str, required=True)
    parser.add_argument('--train_label_atypical', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--exp_root', type=str, default='./experiments')
    parser.add_argument('--model_name', type=str, default='resnet3d')
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to start from (for resuming)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    model_dir = os.path.join(args.exp_root, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.join(model_dir, f"{args.model_name}.pth")
    writer = SummaryWriter(log_dir=os.path.join(args.exp_root, 'logs'))

    transforms_ = transforms.Compose([
        ClipResize((256, 256)),
        RandomCrop(224),
        RandomHorizontalFlip(0.5)
    ])
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    color_jitter = transforms.RandomApply([color_jitter], p=0.8)

    train_dataset_ucf101 = ucf101_pace_pretrain(args.video_dir_ucf101, args.train_label_ucf101, 16, 4, transforms_, color_jitter)
    test_dataset_ucf101 = ucf101_pace_pretrain(args.video_dir_ucf101, args.test_label_ucf101, 16, 4, transforms_, color_jitter)
    train_dataset_atypical = ucf101_pace_pretrain(args.video_dir_atypical, args.train_label_atypical, 16, 4, transforms_, color_jitter)

    print("UCF101 training samples:", len(train_dataset_ucf101))
    print("UCF101 testing samples:", len(test_dataset_ucf101))
    print("Atypical training samples:", len(train_dataset_atypical))

    combined_train_dataset = ConcatDataset([train_dataset_ucf101, train_dataset_atypical])
    ucf101_sample_count = len(train_dataset_ucf101)

    dloader_test = DataLoader(test_dataset_ucf101, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = generate_model(model_depth=50, n_classes=4).to(device)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint)
    print(f"Checkpoint loaded from {args.checkpoint_path}, starting from epoch {args.start_epoch}")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs + 1):
        sampler = get_dynamic_sampler(len(combined_train_dataset), ucf101_sample_count)
        dloader_train_dynamic = DataLoader(combined_train_dataset, sampler=sampler,
                                           batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        train(epoch, model, device, dloader_train_dynamic, optimizer, exp_lr_scheduler, criterion, args, writer)
        acc_record = test(model, device, dloader_test, args)

        if acc_record.avg > best_acc:
            best_acc = acc_record.avg
            torch.save(model.state_dict(), args.model_dir)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{args.model_dir}_epoch_{epoch}.pt")
    writer.close()

if __name__ == '__main__':
    main()
