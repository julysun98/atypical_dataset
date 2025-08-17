
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc, Identity, AverageMeter
from models.resnet3d_supervised import generate_model
from data.ucfdataloader import ucf101
from torch.utils.data import DataLoader
from data.video_transforms import RandomCrop, RandomHorizontalFlip, ClipResize
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os

def train(model, train_loader, labeled_eval_loader, args, writer):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss() 
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            x, label = batch['frames'].to(device, dtype=torch.float), batch['label'].to(device)
            output1, _, _ = model(x)
            loss = criterion(output1, label)
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{args.model_dir}_epoch_{epoch}.pt")
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('Test on labeled classes')
        args.head = 'head1'
        test(model, labeled_eval_loader, args)

def test(model, test_loader, args):
    model.eval() 
    preds, targets = np.array([]), np.array([])
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        x, label = batch['frames'].to(device, dtype=torch.float), batch['label'].to(device)
        output1, output2, _ = model(x)
        output = output1 if args.head == 'head1' else output2
        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())
    acc = cluster_acc(targets.astype(int), preds.astype(int))
    nmi = nmi_score(targets, preds)
    ari = ari_score(targets, preds)
    print('Test acc {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc, nmi, ari))
    return preds

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Supervised Clustering UCF101')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_unlabeled_classes', default=21, type=int)
    parser.add_argument('--num_labeled_classes', default=80, type=int)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset_root', type=str, default='/path/to/ucf101')
    parser.add_argument('--video_dir', type=str, default='/path/to/ucf101/videos')
    parser.add_argument('--train_label_file', type=str, default='/path/to/train_80.txt')
    parser.add_argument('--test_label_file', type=str, default='/path/to/val_80.txt')
    parser.add_argument('--exp_root', type=str, default='./experiments/')
    parser.add_argument('--rotnet_dir', type=str, default='./pretrained/resnet3d.pth')
    parser.add_argument('--model_name', type=str, default='resnet3d_supervised')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir = os.path.join(args.exp_root, runner_name)
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.join(model_dir, '{}.pth'.format(args.model_name))
    writer = SummaryWriter(log_dir=os.path.join(args.exp_root, 'logs/train_supervised'))

    model = generate_model(model_depth=50, n_labeled_classes=args.num_labeled_classes, n_unlabeled_classes=args.num_unlabeled_classes).to(device)
    state_dict = torch.load(args.rotnet_dir)
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)
    model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters():
        if 'head' not in name and 'layer4' not in name:
            param.requires_grad = False

    transform = transforms.Compose([
        ClipResize((256, 256)),
        RandomCrop(224),
        RandomHorizontalFlip(0.5)
    ])

    train_dataset = ucf101(args.video_dir, args.train_label_file, clip_len=16, max_sr=1, transforms_=transform)
    test_dataset = ucf101(args.video_dir, args.test_label_file, clip_len=16, max_sr=4, transforms_=transform)

    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.mode == 'train':
        train(model, train_loader, test_loader, args, writer)
        torch.save(model.state_dict(), args.model_dir)
        print("Model saved to {}.".format(args.model_dir))
    elif args.mode == 'test':
        print("Loading model from {}.".format(args.model_dir))
        model.load_state_dict(torch.load(args.model_dir))
        args.head = 'head1'
        test(model, test_loader, args)
