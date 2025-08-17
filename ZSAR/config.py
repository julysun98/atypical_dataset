import os

class Config:
    # Root directory for datasets
    DATA_ROOT = "dataset"

    # Paths for different datasets
    UCF101_PATH = os.path.join(DATA_ROOT, "UCF-101")
    HMDB51_PATH = os.path.join(DATA_ROOT, "hmdb51", "videos")
    ATYPICAL_PATH = os.path.join(DATA_ROOT, "atypical")
    K400_PATH = os.path.join(DATA_ROOT, "k400")  # Path to Kinetics-400 subset

    # Dataset settings
    DATASET = "HMDB51"  # Options: "UCF101" or "HMDB51"
    NUM_CLASSES = 50  # Total number of classes in the dataset
    NUM_KNOWN_CLASSES = 40  # Number of known classes used for training

    # Pretraining source selection
    # Options include: "ucf", "ucf+atypical", "ucf+k400", "hmdb+k400", etc.
    PRETRAIN_SOURCE = "hmdb+k400"
    PRETRAIN_UCF_RATIO = 0.5  # Ratio of UCF videos used during pretraining

    # Training settings
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    DEVICE = "cuda"  # Options: "cuda" or "cpu"
    SAVE_EVERY = 10  # Save model checkpoint every N epochs

    # Model architecture settings
    FEATURE_DIM = 512
    TEXT_EMBEDDING_DIM = 512
    NUM_FRAMES = 16  # Number of frames sampled from each video
    FRAME_SIZE = 224  # Resized frame dimensions (square)
    HIDDEN_SIZE = 512  # Hidden size for transformer layers
    NUM_LAYERS = 6  # Number of transformer layers
    NUM_HEADS = 8  # Number of attention heads in multi-head attention
    MLP_RATIO = 4  # Expansion factor for MLP layers
    DROPOUT = 0.1  # Dropout probability

    # Checkpoint and logging
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    SPLIT_DIR = "data/splits"

    # Whether to load pretrained video encoder
    LOAD_PRETRAINED = True

    # Dataset split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2

    # Random seed for reproducibility
    SEED = 42

    # Create required directories if they do not exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
