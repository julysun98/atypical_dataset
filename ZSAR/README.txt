ZSAR: Zero-Shot Action Recognition with Pretrained Video-Language Models
===========================================================================

Overview
--------
ZSAR is a framework for zero-shot action recognition using pretrained video-language models.
It supports training on known classes and evaluating on unseen (unknown) classes using a prompt-based approach.

This implementation includes:

- Pretraining using contrastive learning (`pretrain.py`)
- Zero-shot recognition training using similarity matching (`train.py`)
- Final evaluation on unseen classes (`evaluate.py`)
- Support for datasets like UCF101, HMDB51, Kinetics-400, and Atypical videos

Quick Start
-----------

1. Clone the repository and set up the environment:

    ```bash
    conda env create -f environment.yml
    conda activate ZSAR
    ```

2. Organize your data directory:

    ```
    dataset/
    ├── UCF-101/
    ├── hmdb51/videos/
    ├── atypical/videos
    └── k400/
    ```

3. Run pretraining (optional but recommended):

    ```bash
    python pretrain.py
    ```

4. Train the ZSAR model on known classes:

    ```bash
    python train.py
    ```

5. Evaluate on unseen (unknown) classes:

    ```bash
    python evaluate.py
    ```

Configuration
-------------
All experiment settings are managed in `config.py`. Key parameters include:

- `PRETRAIN_SOURCE`: determines which datasets are used for pretraining
- `DATASET`: choose between 'UCF101' or 'HMDB51'
- `LOAD_PRETRAINED`: whether to load pretrained video encoder
- `CHECKPOINT_DIR`: directory to save models
- `NUM_KNOWN_CLASSES`: number of base classes used for training
- `NUM_EPOCHS`, `BATCH_SIZE`, etc. for training configuration

File Structure
--------------
- `pretrain.py`        - Contrastive video-language pretraining
- `train.py`           - Zero-shot action recognition training on base classes
- `evaluate.py`        - Evaluation on unseen classes
- `models/`            - Model definitions (CLIP encoder, ZSAR logic)
- `data/`              - Dataset loading and splitting utilities
- `config.py`          - All configuration values
