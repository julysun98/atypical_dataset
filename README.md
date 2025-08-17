## 🔍 Overview

The **Atypical Dataset** is a curated benchmark designed to advance **open-world learning**, particularly in **Out-of-Distribution (OOD) detection**, **Novel Class Discovery (NCD)**, and **Zero-Shot Action Recognition (ZSAR)**.

Our dataset provides challenging, low-frequency, and conceptually diverse video samples that simulate atypical cases encountered in open environments. The dataset has been accepted to **BMVC 2025**. You can read the full paper [here](<>).

Dataset access:  
📦 [Hugging Face Dataset Page](https://huggingface.co/datasets/mixgroup-atypical/atypical)

---

## 📁 Dataset Structure

The `atypical` dataset complements four existing action recognition datasets:

- [Kinetics-400](https://www.deepmind.com/open-source/kinetics)
- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- [MiT-v2]([https://github.com/raoyongming/MiT](http://moments.csail.mit.edu))
- Download our `atypical` dataset from Hugging Face:  
   [https://huggingface.co/datasets/mixgroup-atypical/atypical](https://huggingface.co/datasets/mixgroup-atypical/atypical)

You will need to download these datasets in addition to the atypical samples to run our experiments.

---

## 🔬 Supported Tasks

We demonstrate the effectiveness of the **Atypical Dataset** on the following open-world learning tasks:

### 1. Out-of-Distribution (OOD) Detection  
   - Detect samples not seen during training.
   - See usage instructions in: `OODDetection/README.txt`

### 2. Novel Class Discovery (NCD)  
   - Identify new categories from unlabeled data.
   - See usage instructions in: `NCD/README.txt`

### 3. Zero-Shot Action Recognition (ZSAR)  
   - Recognize actions not seen during training by leveraging semantic information.
   - See usage instructions in: `ZSAR/README.txt`

Each task has its own folder containing experiment setups, model configurations, and evaluation scripts.

---


## 📊 Citation

If you use our dataset in your research, please cite our BMVC 2025 paper:

```bibtex
@inproceedings{BMVC2025atypical,
  title={What Can We Learn from Harry Potter? An Exploratory Study of Visual Representation Learning from
Atypical Videos},
author={Qiyue Sun, Qiming Huang, Yang Yang, Hongjun Wang, Jianbo Jiao},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2025}
}
