import os
import json
import random
import numpy as np
from config import Config

def create_class_splits():
    """
    Create splits for ZSAR:
    - Pretrain: use all classes
    - Train: use known classes (e.g., 80 for UCF101 or 40 for HMDB51)
    - Test: use remaining (unknown) classes
    """
    dataset_path = Config.UCF101_PATH if Config.DATASET == "UCF101" else Config.HMDB51_PATH
    all_classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print(all_classes)
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)

    known_classes = set(random.sample(all_classes, Config.NUM_KNOWN_CLASSES))
    unknown_classes = set(all_classes) - known_classes

    splits = {
        'pretrain': {'all_classes': all_classes},
        'train': {'known_classes': sorted(list(known_classes))},
        'test': {'unknown_classes': sorted(list(unknown_classes)), 'all_classes': all_classes}
    }

    os.makedirs(Config.SPLIT_DIR, exist_ok=True)
    split_file = os.path.join(Config.SPLIT_DIR, f'{Config.DATASET.lower()}_splits.json')
    with open(split_file, 'w') as f:
        json.dump(splits, f, indent=4)

    return splits

def load_splits():
    split_file = os.path.join(Config.SPLIT_DIR, f'{Config.DATASET.lower()}_splits.json')
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            return json.load(f)
    else:
        return create_class_splits()

def get_split_videos(split_type='pretrain'):
    splits = load_splits()
    base_dataset = Config.DATASET
    base_path = Config.UCF101_PATH if base_dataset == "UCF101" else Config.HMDB51_PATH
    videos = []

    if split_type == 'pretrain':
        base_videos, atypical_videos, k400_videos = [], [], []
        base_classes = splits['pretrain']['all_classes']

        # Collect base videos
        for class_name in base_classes:
            class_dir = os.path.join(base_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.endswith(('.avi', '.mp4')):
                    video_path = os.path.join(class_dir, file)
                    base_videos.append((video_path, class_name))

        num_base = len(base_videos)

        # Atypical videos
        if 'atypical' in Config.PRETRAIN_SOURCE:
            atypical_root = os.path.join(Config.ATYPICAL_PATH, 'videos')
            if os.path.exists(atypical_root):
                for style_class in os.listdir(atypical_root):
                    style_dir = os.path.join(atypical_root, style_class)
                    if not os.path.isdir(style_dir):
                        continue
                    for file in os.listdir(style_dir):
                        if file.endswith(('.avi', '.mp4')):
                            video_path = os.path.join(style_dir, file)
                            atypical_videos.append((video_path, 'none'))

        # Kinetics-400 videos
        if 'k400' in Config.PRETRAIN_SOURCE:
            k400_root = os.path.join(Config.K400_PATH, "videos", "train_256")
            for root, _, files in os.walk(k400_root):
                for file in files:
                    if file.endswith(('.avi', '.mp4')):
                        video_path = os.path.join(root, file)
                        k400_videos.append((video_path, 'none'))
            print(f"[INFO] K400 videos found: {len(k400_videos)}")

        # Compose pretrain videos
        ratio_base = Config.PRETRAIN_UCF_RATIO
        if Config.PRETRAIN_SOURCE in ['ucf', 'hmdb']:
            videos = base_videos
        elif 'atypical' in Config.PRETRAIN_SOURCE:
            n_base = int(num_base * ratio_base)
            n_atypical = num_base - n_base
            videos = random.sample(base_videos, n_base)
            videos += random.sample(atypical_videos, min(len(atypical_videos), n_atypical))
        elif 'k400' in Config.PRETRAIN_SOURCE:
            n_base = int(num_base * ratio_base)
            n_k400 = num_base - n_base
            videos = random.sample(base_videos, n_base)
            videos += random.sample(k400_videos, min(len(k400_videos), n_k400))

        random.shuffle(videos)
        return videos

    # Train/Test
    classes = splits['train']['known_classes'] if split_type == 'train' else splits['test']['unknown_classes']

    for class_name in classes:
        class_dir = os.path.join(base_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for video_file in os.listdir(class_dir):
            if video_file.endswith(('.avi', '.mp4')):
                video_path = os.path.join(class_dir, video_file)
                videos.append((video_path, class_name))

    random.shuffle(videos)
    return videos
