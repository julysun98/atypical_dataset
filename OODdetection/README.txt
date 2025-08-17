Overview:
---------
This project implements open-set video classification techniques using Outlier Exposure (OE). It provides tools for training models from scratch, fine-tuning pre-trained models, evaluating performance, and comparing against a baseline. The project uses a ResNet backbone and supports multi-channel video inputs (e.g., RGB, optical flow).

Project Structure:
------------------
- `baseline.py`  
  Trains a standard video classification model (without OE), using cross-entropy loss. Serves as the baseline.

- `oe_scratch.py`  
  Trains a model from scratch with OE. Incorporates outlier samples into the training process to improve open-set robustness.

- `oe_tune.py`  
  Fine-tunes a pre-trained model using OE, making it more robust to unknown or out-of-distribution inputs.

- `models/resnet.py`  
  Implements the ResNet architecture used throughout the project. Supports different depths (e.g., ResNet-18), and input channels.

- `test.py`  
  Evaluates trained models on held-out test data. Computes classification accuracy, AUROC, and other open-set metrics.

- `videodataset.py`  
  Custom PyTorch Dataset class for loading and preprocessing video data. Handles per-frame loading, label mapping, and channel selection.

Usage Instructions:
-------------------

1. **Data Preparation**
   - Organize video data into folders by class (e.g., `train/class_x/*.jpg`).
   - For OE training, prepare a separate dataset of "outlier" examples.
   - Modify `videodataset.py` if necessary to match your data layout.

2. **Train a Baseline Model**
   ```
   python baseline.py
   ```

3. **Train with OE from Scratch**
   ```
   python oe_scratch.py
   ```

4. **Fine-tune a Pre-trained Model with OE**
   ```
   python oe_tune.py
   ```

5. **Evaluate a Model**
   ```
   python test.py -m oe_tune
   ```

Requirements:
-------------
- Python 3.7+
- PyTorch >= 1.8
- torchvision
- numpy, tqdm, scikit-learn, Pillow

Metrics:
--------
- **Top-1 Accuracy** – Classification accuracy on known classes.
- **AUROC** – Open-set detection metric.
- **FPR@95TPR** – False Positive Rate at 95% True Positive Rate.
