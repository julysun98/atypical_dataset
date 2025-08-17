import os
import sys
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from models.resnet import generate_model
from videodataset import VideoDataset

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures

parser = argparse.ArgumentParser(description='Evaluate video OOD detector')
parser.add_argument('--test_bs', type=int, default=1)
parser.add_argument('--num_to_avg', type=int, default=5)
parser.add_argument('--use_xent', '-x', action='store_true')
parser.add_argument('--method_name', '-m', type=str, default='baseline')
parser.add_argument('--load', '-l', type=str, default='./snapshots')
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=2)
parser.add_argument('--data_root', type=str, default='/path/to/data')
args = parser.parse_args()

# ==== Paths ====
UCF101_VIDEO_DIR = '/path/to/ucf101/videos'
UCF101_LABEL_FILE = '/path/to/ucf101/labels/filtered_ucf101_val_split_1_videos.txt'

HMDB51_VIDEO_DIR = '/path/to/hmdb51/videos'
HMDB51_LABEL_FILE = '/path/to/hmdb51/labels/filtered_hmdb51_val_split_1_videos.txt'

MIT_VIDEO_DIR = '/path/to/mit/videos'
MIT_LABEL_FILE = '/path/to/mit/labels/filtered_mit_val_split_1_videos_new.txt'

THEATER_VIDEO_DIR = '/path/to/atypical/videos'
THEATER_LABEL_FILE = '/path/to/atypical/labels/theater_video_list.txt'

SURREAL_VIDEO_DIR = '/path/to/surreal/videos'
SURREAL_LABEL_FILE = '/path/to/surreal/surreal_video_list.txt'

# ==== Transforms ====
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== Load in-distribution test set ====
test_data = VideoDataset(UCF101_VIDEO_DIR, UCF101_LABEL_FILE, clip_len=8, frame_interval=8, num_clips=1, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch)

# ==== Load model ====
net = generate_model(model_depth=50, n_classes=93)
model_path = os.path.join(args.load, args.method_name, f'{args.method_name}_epoch_5.pt')
assert os.path.isfile(model_path), f'Model not found at {model_path}'
net.load_state_dict(torch.load(model_path))
net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
if args.ngpu > 0:
    net.cuda()

cudnn.benchmark = True

to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)

def get_ood_scores(loader, in_dist=False):
    scores, right_scores, wrong_scores = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            data = batch['frames']
            target = batch['label']
            if torch.cuda.is_available():
                data = data.cuda(non_blocking=True)

            _, output = net(data)
            softmax_output = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                score = to_np(output.mean(1) - torch.logsumexp(output, dim=1))
            else:
                score = -np.max(softmax_output, axis=1)

            scores.append(score)

            if in_dist:
                preds = np.argmax(softmax_output, axis=1)
                targets = target.numpy().squeeze()
                correct = preds == targets
                incorrect = ~correct

                right_scores.append(score[correct])
                wrong_scores.append(score[incorrect])

    if in_dist:
        return concat(scores), concat(right_scores), concat(wrong_scores)
    return concat(scores)

in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)
num_right = len(right_score)
num_wrong = len(wrong_score)
print(f'Error Rate: {100.0 * num_wrong / (num_right + num_wrong):.2f}%')
print('\n[In-Distribution] Error Detection:')
show_performance(wrong_score, right_score, method_name=args.method_name)

auroc_list, aupr_list, fpr_list = [], [], []

def get_and_print_results(name, video_dir, label_file):
    ood_data = VideoDataset(video_dir, label_file, clip_len=8, frame_interval=8, num_clips=1, transform=test_transform)
    ood_loader = DataLoader(ood_data, batch_size=args.test_bs, shuffle=True, num_workers=args.prefetch)

    aurocs, auprs, fprs = [], [], []
    for _ in range(args.num_to_avg):
        out_score = get_ood_scores(ood_loader)
        auroc, aupr, fpr = get_measures(out_score, in_score)
        aurocs.append(auroc)
        auprs.append(aupr)
        fprs.append(fpr)

    print(f'\n[OOD] {name} Detection:')
    print_measures(np.mean(aurocs), np.mean(auprs), np.mean(fprs), method_name=args.method_name)
    auroc_list.append(np.mean(aurocs))
    aupr_list.append(np.mean(auprs))
    fpr_list.append(np.mean(fprs))

get_and_print_results('HMDB51', HMDB51_VIDEO_DIR, HMDB51_LABEL_FILE)
get_and_print_results('MIT', MIT_VIDEO_DIR, MIT_LABEL_FILE)
get_and_print_results('Theater', THEATER_VIDEO_DIR, THEATER_LABEL_FILE)
get_and_print_results('Surreal', SURREAL_VIDEO_DIR, SURREAL_LABEL_FILE)

print('\n\nMean Test Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
