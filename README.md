# 🦅 Bird Species Classification Pipeline - DRACCAR Project

**A modular PyTorch pipeline for identifying bird species in offshore wind farm environments.**

> **Context:** This project was realized as part of the curriculum at **IMT Atlantique**, in collaboration with **France Energies Marines**. It contributes to the broader **DRACCAR-MMERMAID** initiative, aiming to evaluate the ecological impact of offshore wind farms on avian populations.

---

## 📖 Overview

This repository contains the **classification module** of the DRACCAR pipeline. It is designed to process cropped images from surveillance cameras and identify bird species, even with highly imbalanced datasets.

The system is built on a **modular PyTorch architecture** optimized for:
* **Reproducibility:** Strict seeding and configuration management.
* **Efficiency:** Custom "Tensor-in-RAM" caching to maximize GPU throughput.
* **Flexibility:** Seamless integration with **HuggingFace Datasets** and local files.
* **Cross-Platform Support:** Native support for **CUDA** (Nvidia) and **DirectML** (AMD/Windows).

## 📂 Project Structure

The source code is organized by responsibility to ensure modularity and ease of integration into notebooks (Kaggle/Colab).

| Module | Component | Description |
| :--- | :--- | :--- |
| **`data/`** | `loaders.py` | Custom loaders (`HuggingFaceImageDatasetLoader`) and balancing logic. |
| | `datasets.py` | Implements **`TensorInRamDataset`** to pre-load and cache tensors, removing CPU bottlenecks. |
| **`preprocessing/`** | `pipeline.py` | Manages transformation pipelines (`TransformPipeline`) for training and evaluation. |
| **`model/`** | `train.py` | Contains the `Trainer` and `TrainingStage` classes for orchestrated learning. |
| | `test.py` | Handles inference, evaluation metrics, and automatic threshold calibration. |
| **`utils/`** | `config.py` | Centralized configuration management. |

---

## 🚀 Key Technical Implementations

### 1. Optimized Data Pipeline ("Tensor-In-RAM")
Standard data loaders often bottleneck the GPU by processing raw images on the CPU during training.
* **Solution:** This pipeline converts images to tensors and applies invariant transforms (resize, normalize) *once* at startup, caching them in RAM.
* **Benefit:** Training shifts solely to dynamic augmentations (flips, jitters), significantly increasing GPU utilization.

### 2. Staged Training Strategy
To handle Transfer Learning and class imbalance effectively, the pipeline easily allows for multi-stage training through the `TrainingStage` class. Here is an example used during our project :
* **Phase 1 (Stabilization):** Backbone is **frozen**. Training focuses on the classification head using **Focal Loss** and a **Balanced Sampler** to learn rare classes without destroying pre-trained feature maps.
* **Phase 2 (Fine-Tuning):** Backbone is **progressively unfrozen**. The loss switches to **CrossEntropy**, and a `ReduceOnPlateau` scheduler fine-tunes the entire network.

---

## 🛠️ Installation

The pipeline is designed to be installed directly from the source repository.

~~~bash
# Install via pip/git
pip install git+https://github.com/PierreAHignard/Modular-Pipeline-Source.git
~~~

---

## 💻 Usage Example

This pipeline is optimized for use within Jupyter Notebooks (e.g., Kaggle). Below is a standard workflow.

### 1. Configuration & Data Loading
Define your `TRAINING_CONFIG` and initialize the loader.

~~~python
from data.loaders import HuggingFaceImageDatasetLoader
from preprocessing.pipeline import TransformPipeline
from utils.config import Config

# 1. Setup Configuration
TRAINING_CONFIG = {
    'dataset.id': "[your hf name]/[dataset name]",
    'dataset.only_specified_labels': False,
    'output_dir': Path("/kaggle/working"),
    # ... other params
}
config = Config(TRAINING_CONFIG, ...)

# 2. Initialize Transforms & Loader
train_transforms = TransformPipeline(config=config, model_name='resnet50', is_train=True)

train_loader = HuggingFaceImageDatasetLoader(
    dataset_name=TRAINING_CONFIG["dataset.id"],
    split="train",
    cache_dir=TRAINING_CONFIG["output_dir"] / "working" / "cache",
    batch_size=64,
    transforms=train_transforms,
    config=config,
    label_column="labels"
)
~~~

### 2. Defining Training Stages
Configure the **Staged Training** strategy to handle imbalance.

~~~python
from model.train import TrainingStage, Trainer
from model.utils import FocalLoss
from data.utils import create_balanced_loader
import torch.nn as nn

# Phase 1: Balanced Classes + Focal Loss (Backbone Frozen)
stage1 = TrainingStage(
    name="Phase 1 - Stabilization",
    epochs=20,
    learning_rate=1e-3,
    criterion=FocalLoss(alpha=1, gamma=2),
    train_loader=create_balanced_loader(train_loader._dataset, batch_size=32),
    unfreeze_schedule=[], # Frozen backbone
    classifier_lr_multiplier=10.0
)

# Phase 2: Full Fine-Tuning (Backbone Unfrozen)
stage2 = TrainingStage(
    name="Phase 2 - Fine-Tuning",
    epochs=30,
    learning_rate=1e-4,
    criterion=nn.CrossEntropyLoss(),
    train_loader=None, # Uses default loader
    unfreeze_schedule=[5, 10], # Progressive unfreeze
    classifier_lr_multiplier=5.0
)
~~~

### 3. Execution
Instantiate the `Trainer` and fit the model.

~~~python
trainer = Trainer(
    model=model,
    train_loader=train_dataloader,
    val_loader=eval_dataloader,
    config=config,
    device=device,
    training_stages=[stage1, stage2]
)

history = trainer.fit()
~~~

---

## 📊 Evaluation

Run full evaluation metrics on the test set, including confusion matrices and per-class accuracy.

~~~python
from model.test import Tester

tester = Tester(model, config)
results = tester.run_full_evaluation(test_dataloader)
~~~

---

## 📜 Credits

* **Author:** Pierre-Antoine HIGNARD
* **Organization:** IMT Atlantique & France Energies Marines
* **Project:** DRACCAR-MMERMAID