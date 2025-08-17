# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from models.resnet import generate_model
from videodataset import VideoDataset

writer = SummaryWriter(log_dir='runs/experiment2')

# Optional: import validation split
if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.validation_dataset import validation_split

# Argument parser
parser = argparse.ArgumentParser(description='Train with Outlier Exposure from scratch')
parser.add_argument('--calibration', '-c', action='store_true')
parser.add_argument('--epochs', '-e', type=int, default=100)
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1)
parser.add_argument('--batch_size', '-b', type=int, default=16)
parser.add_argument('--oe_batch_size', type=int, default=16)
parser.add_argument('--test_bs', type=int, default=16)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', '-d', type=float, default=0.0005)
parser.add_argument('--save', '-s', type=str, default='./snapshots/oe_scratch')
parser.add_argument('--load', '-l', type=str, default='')
parser.add_argument('--test', '-t', action='store_true')
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=4)
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)

# === Dataset paths (replace with your own) ===
VIDEO_DIR_IN = '/path/to/ucf101/videos'
VIDEO_DIR_OUT = '/path/to/kinetics/videos'
LABEL_FILE_IN = '/path/to/ucf101_train_labels.txt'
LABEL_FILE_OUT = '/path/to/kinetics_train_labels.txt'
LABEL_FILE_TEST = '/path/to/ucf101_val_labels.txt'
NUM_CLASSES = 93

# === Transforms ===
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Dataset & DataLoaders ===
train_dataset_in = VideoDataset(VIDEO_DIR_IN, LABEL_FILE_IN, clip_len=8, frame_interval=8, num_clips=1, transform=train_transform)
train_loader_in = DataLoader(train_dataset_in, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch)

train_dataset_out = VideoDataset(VIDEO_DIR_OUT, LABEL_FILE_OUT, clip_len=8, frame_interval=8, num_clips=1, transform=train_transform)
train_loader_out = DataLoader(train_dataset_out, batch_size=args.oe_batch_size, shuffle=True, num_workers=args.prefetch)

test_dataset = VideoDataset(VIDEO_DIR_IN, LABEL_FILE_TEST, clip_len=8, frame_interval=8, num_clips=1, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch)

# === Model ===
net = generate_model(model_depth=50, n_classes=NUM_CLASSES)
start_epoch = 0

# Load checkpoint
if args.load:
    for i in range(999, -1, -1):
        model_path = os.path.join(args.load, f'oe_scratch_epoch_{i}.pt')
        if os.path.isfile(model_path):
            net.load_state_dict(torch.load(model_path))
            print(f'Model restored from epoch {i}')
            start_epoch = i + 1
            break
    else:
        raise FileNotFoundError("No valid checkpoint found.")

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True

optimizer = torch.optim.SGD(
    net.parameters(), lr=args.learning_rate,
    momentum=args.momentum, weight_decay=args.decay, nesterov=True
)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,
        1e-6 / args.learning_rate
    )
)

global_step = 0

def train():
    global global_step
    net.train()
    loss_avg = 0.0

    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))

    for in_set, out_set in zip(train_loader_in, train_loader_out):
        data_in, data_out = in_set['frames'], out_set['frames']
        data = torch.cat((data_in, data_out), dim=0)
        target = in_set['label']

        data, target = data.cuda(), target.cuda()

        _, logits = net(data)
        logits_in = logits[:len(data_in)]
        logits_out = logits[len(data_in):]

        loss_in = F.cross_entropy(logits_in, target)
        reg_term = 0.5 * -(logits_out.mean(1) - torch.logsumexp(logits_out, dim=1)).mean()
        loss = loss_in + reg_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_avg = loss_avg * 0.8 + loss.item() * 0.2

        writer.add_scalar('Loss/train_ce', loss_in.item(), global_step)
        writer.add_scalar('Loss/regularization', reg_term.item(), global_step)
        writer.add_scalar('Loss/total', loss.item(), global_step)

        global_step += 1

    state['train_loss'] = loss_avg

def test():
    net.eval()
    loss_total = 0.0
    correct_top1 = 0
    correct_top5 = 0
    global global_step

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            data, target = batch['frames'], batch['label']
            data, target = data.cuda(), target.cuda()

            _, logits = net(data)
            loss = F.cross_entropy(logits, target)
            loss_total += loss.item()

            pred_top1 = logits.argmax(1, keepdim=True)
            correct_top1 += pred_top1.eq(target.view_as(pred_top1)).sum().item()

            _, pred_top5 = logits.topk(5, 1, True, True)
            pred_top5 = pred_top5.t()
            correct_top5 += pred_top5.eq(target.view(1, -1).expand_as(pred_top5)).sum().item()

            writer.add_scalar('Test/loss', loss_total / (idx + 1), global_step)
            writer.add_scalar('Test/accuracy_top1', correct_top1 / len(test_loader.dataset), global_step)
            writer.add_scalar('Test/accuracy_top5', correct_top5 / len(test_loader.dataset), global_step)

            global_step += 1

    total = len(test_loader.dataset)
    state['test_loss'] = loss_total / len(test_loader)
    state['test_accuracy_top1'] = correct_top1 / total
    state['test_accuracy_top5'] = correct_top5 / total
    state['test_accuracy'] = state['test_accuracy_top1']

    print(f"Test Loss: {state['test_loss']:.4f}, Top-1 Acc: {state['test_accuracy_top1']:.4f}, Top-5 Acc: {state['test_accuracy_top5']:.4f}")

if args.test:
    test()
    print(state)
    exit()

# === Training Loop ===
os.makedirs(args.save, exist_ok=True)
log_file = os.path.join(args.save, 'oe_scratch_training_results.csv')
with open(log_file, 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')
save_interval = 10

for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch
    start_time = time.time()

    train()
    test()

    if (epoch + 1) % save_interval == 0:
        ckpt_path = os.path.join(args.save, f'oe_scratch_epoch_{epoch + 1}.pt')
        torch.save(net.state_dict(), ckpt_path)
        print(f'Checkpoint saved: {ckpt_path}')

    with open(log_file, 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            epoch + 1,
            int(time.time() - start_time),
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy']
        ))

    print('Epoch {:03d} | Time {:5d}s | Train Loss {:.4f} | Test Loss {:.4f} | Test Error {:.2f}%'.format(
        epoch + 1,
        int(time.time() - start_time),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy']
    ))

writer.close()
