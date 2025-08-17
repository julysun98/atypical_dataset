# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.resnet import generate_model
from videodataset import VideoDataset

# Setup TensorBoard writer
writer = SummaryWriter(log_dir='runs/baseline')

# Import validation split function
if __package__ is None:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.validation_dataset import validation_split

# Argument parser
parser = argparse.ArgumentParser(description='Train a video classification model')
parser.add_argument('--calibration', '-c', action='store_true', help='Use validation split for calibration.')
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of training epochs.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=16, help='Training batch size.')
parser.add_argument('--test_bs', type=int, default=16, help='Test batch size.')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum.')
parser.add_argument('--save', '-s', type=str, default='./snapshots/baseline', help='Checkpoint save path.')
parser.add_argument('--load', '-l', type=str, default='', help='Path to load checkpoint.')
parser.add_argument('--test', '-t', action='store_true', help='Test-only mode.')
parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs (0 = CPU).')
parser.add_argument('--prefetch', type=int, default=2, help='DataLoader prefetch threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

# Set random seed
torch.manual_seed(1)
np.random.seed(1)

# Dataset paths
video_dir = '/your/path/to/videos'  # Replace with your actual path
train_label_file = '/your/path/to/train_labels.txt'
test_label_file = '/your/path/to/test_labels.txt'
num_classes = 93

# Transformations
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = VideoDataset(video_dir, train_label_file, clip_len=8, frame_interval=8, num_clips=1, transform=train_transform)
test_dataset = VideoDataset(video_dir, test_label_file, clip_len=8, frame_interval=8, num_clips=1, transform=test_transform)

if args.calibration:
    train_dataset, val_dataset = validation_split(train_dataset, val_share=0.1)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False, num_workers=4)

# Create model
net = generate_model(model_depth=50, n_classes=num_classes)
start_epoch = 0

# Load checkpoint if provided
if args.load:
    for i in range(999, -1, -1):
        model_path = os.path.join(args.load, f'baseline_epoch_{i}.pt')
        if os.path.isfile(model_path):
            net.load_state_dict(torch.load(model_path))
            print(f'Model restored from {model_path}')
            start_epoch = i + 1
            break

# Move model to GPU if available
if args.ngpu > 1:
    net = nn.DataParallel(net, device_ids=list(range(args.ngpu)))
if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True

optimizer = torch.optim.SGD(
    net.parameters(), lr=state['learning_rate'],
    momentum=state['momentum'], weight_decay=0.0005, nesterov=True
)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step, args.epochs * len(train_loader), 1, 1e-6 / args.learning_rate
    )
)

global_step = 0

def train():
    global global_step
    net.train()
    loss_avg = 0.0
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        data = batch['frames']
        target = batch['label']
        if torch.cuda.is_available():
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        _, logits = net(data)
        loss = F.cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        writer.add_scalar('Loss/train', loss.item(), global_step)
        writer.add_scalar('Loss/avg', loss_avg, global_step)
        global_step += 1

    state['train_loss'] = loss_avg

def test():
    net.eval()
    loss_total = 0.0
    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            data = batch['frames']
            target = batch['label']
            if torch.cuda.is_available():
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            _, output = net(data)
            loss = F.cross_entropy(output, target)
            loss_total += loss.item()

            pred_top1 = output.argmax(1, keepdim=True)
            correct_top1 += pred_top1.eq(target.view_as(pred_top1)).sum().item()

            _, pred_top5 = output.topk(5, dim=1)
            pred_top5 = pred_top5.t()
            correct_top5 += pred_top5.eq(target.view(1, -1).expand_as(pred_top5)).sum().item()

    total_samples = len(test_loader.dataset)
    state['test_loss'] = loss_total / len(test_loader)
    state['test_accuracy_top1'] = correct_top1 / total_samples
    state['test_accuracy_top5'] = correct_top5 / total_samples
    state['test_accuracy'] = state['test_accuracy_top1']

    print(f"Test Loss: {state['test_loss']:.4f}, Top-1 Acc: {state['test_accuracy_top1']:.4f}, Top-5 Acc: {state['test_accuracy_top5']:.4f}")

# Test-only mode
if args.test:
    test()
    print(state)
    exit()

# Make save directory
os.makedirs(args.save, exist_ok=True)
if not os.path.isdir(args.save):
    raise Exception(f'{args.save} is not a directory')

result_path = os.path.join(args.save, 'baseline_training_results.csv')
with open(result_path, 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%),test_top1,test_top5\n')

print('Beginning Training\n')

save_interval = 10

for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch
    start_time = time.time()

    train()
    test()

    if (epoch + 1) % save_interval == 0:
        ckpt_path = os.path.join(args.save, f'baseline_epoch_{epoch+1}.pt')
        torch.save(net.state_dict(), ckpt_path)
        print(f'Checkpoint saved: {ckpt_path}')

    with open(result_path, 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f,%0.6f,%0.6f\n' % (
            epoch + 1,
            time.time() - start_time,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
            state['test_accuracy_top1'],
            state['test_accuracy_top5']
        ))

    print('Epoch {0:3d} | Time {1:5d}s | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}%'.format(
        epoch + 1,
        int(time.time() - start_time),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy']
    ))
