# AutoNovel Adaptation for UCF101

This repository adapts the [AutoNovel](http://www.robots.ox.ac.uk/~vgg/research/auto_novel/) framework for video classification using the atypical dataset.

Authors of original work:
- Kai Han*, Sylvestre-Alvise Rebuffi*, Sebastien Ehrhardt*, Andrea Vedaldi, Andrew Zisserman  
  (* indicates equal contribution)  
  [AutoNovel Paper – ICLR 2020](https://openreview.net/forum?id=H1gNOeHKDS)

---

## Project Overview

This project consists of three stages:

1. **Self-supervised learning**  
   Trains a model using rotation-based self-supervised learning on atypical/kinetics/ucf101 data.  
   → `selfsupervised_learning.py`

2. **Supervised learning**  
   Fine-tunes the model on labeled data using cross-entropy loss.  
   → `supervised_learning.py`

3. **Novel category discovery (AutoNovel)**  
   Jointly trains the model to classify known (labeled) classes and discover new (unlabeled) categories using ranking statistics.  
   → `auto_novel.py`

---

## Folder Structure

.
├── selfsupervised_learning.py       # Step 1  
├── supervised_learning.py           # Step 2  
├── auto_novel.py                    # Step 3  
├── data/  
│   ├── datasets/                    # Put atypical split files here  
│   ├── experiments/                 # Checkpoints and logs  
├── models/                          # Network definitions  
├── utils/                           # Losses, tools, transforms  
└── logs/                            # TensorBoard logs  

---

## Dependencies

Install required dependencies (recommended via Conda):

    conda env create -f environment.yml
    conda activate auto_novel

---

## Dataset

We use the **atypical** video action dataset. Your directory structure should be:

/your/dataset/path/  
├── videos/                         # Raw video clips  
├── ucf101_train_80.txt            # Labeled training split (80 classes)  
├── ucf101_train_21.txt            # Unlabeled training split (21 classes)  
├── ucf101_val_80.txt              # Labeled validation split  
├── ucf101_val_21.txt              # Unlabeled validation split  

Specify paths in the corresponding script via `--dataset_root` and `--exp_root`.

---

## Training Steps

### 1. Self-Supervised Pretraining

    python selfsupervised_learning.py

### 2. Supervised Fine-Tuning

    python supervised_learning.py

### 3. AutoNovel Training

    python auto_novel.py

To run evaluation only:

    python auto_novel.py --mode test

---

## Evaluation

The script automatically evaluates on:
- Labeled classes (head1)
- Unlabeled classes (head2)

Metrics:
- Accuracy
- Normalized Mutual Information (NMI)
- Adjusted Rand Index (ARI)

---


This implementation is based on the original [AutoNovel](https://github.com/zhufeifan/auto_novel) paper and codebase, extended to support the atypical dataset for video classification tasks.
