# DeepIFSAC

Official implementation of **"DeepIFSAC: Deep Imputation of Missing Values Using Feature and Sample Attention within Contrastive Framework"**.

---

## Paper

If you use this repository, please cite our paper:

> Kowsar, I., Rabbani, S. B., Hou, Y., & Samad, M. D. (2025).
> **DeepIFSAC: Deep imputation of missing values using feature and sample attention within contrastive framework.**
> *Knowledge-Based Systems, 318,* 113506.
> [https://doi.org/10.1016/j.knosys.2025.113506](https://doi.org/10.1016/j.knosys.2025.113506)

```bibtex
@article{kowsar2025deepifsac,
  title={DeepIFSAC: Deep imputation of missing values using feature and sample attention within contrastive framework},
  author={Kowsar, Ibna and Rabbani, Shourav B and Hou, Yina and Samad, Manar D},
  journal={Knowledge-Based Systems},
  volume={318},
  pages={113506},
  year={2025},
  publisher={Elsevier}
}
```

**DeepIFSAC** is a deep learning framework for tabular data that combines attention-based architecture with contrastive learning for missing value imputation. It supports both OpenML benchmark datasets and arbitrary numpy/pandas inputs via a sklearn-compatible API.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Training the Model](#training-the-model)
5. [Results and Outputs](#results-and-outputs)
6. [sklearn-style Imputer API](#sklearn-style-imputer-api)
7. [Project Structure](#project-structure)

---

## Overview

DeepIFSAC implements a **pretraining phase** (contrastive and denoising objectives with CutMix/MixUp augmentation) and a **finetuning phase** for downstream classification tasks.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/eyshin05/DeepIFSAC.git
cd DeepIFSAC
```

### Create a Virtual Environment with uv

```bash
uv venv .venv --python 3.11
uv sync
source .venv/bin/activate
```

To consume DeepIFSAC directly from Git in another uv-managed project:

```bash
uv add "deepifsac @ git+https://github.com/mdsamad001/DeepIFSAC.git"
```

Alternatively, using conda:

```bash
conda env create -f difsac_env.yml
conda activate DeepIFSAC
```

---

## Dataset

The training script uses `my_data_prep_openml` from the `data_openml` module. Provide a dataset ID via `--dset_id` to automatically download and preprocess the dataset from OpenML.

---

## Training the Model

Training has two phases: **pretraining (imputation)** and **finetuning (classification)**.

### Pretraining

Pretraining uses denoising and/or contrastive objectives. Run with the `--pretrain` flag:

```bash
python my_train.py \
  --dset_id 11 \
  --task multiclass \
  --attentiontype colrow \
  --pretrain \
  --pretrain_epochs 1000 \
  --epochs 200 \
  --batchsize 128 \
  --dset_seed 0 \
  --cuda_device 0 \
  --use_default_model \
  --missing_rate 0.5 \
  --missing_type mcar \
  --pt_aug cutmix
```

- `--attentiontype colrow` → DeepIFSAC with contrastive loss (`pt_tasks = ['denoising', 'contrastive']`)
- `--attentiontype colrowatt` → DeepIFSAC without contrastive loss (`pt_tasks = ['denoising']`)

```bash
python my_train.py --help
```

### Finetuning / Downstream Evaluation

After pretraining, the model can be evaluated using classical classifiers (Logistic Regression, Gradient Boosting) on extracted features, or finetuned with a separate MLP head.

---

## Results and Outputs

| Output | Location |
|--------|----------|
| Model weights | `./results/model_weights/` |
| Training metrics | `./results/training_scores/` (pickle) |
| Downstream performance | printed to console |

---

## sklearn-style Imputer API

A sklearn-compatible imputer class for arbitrary numpy/pandas input — no OpenML dependency required.

### Usage

```python
import numpy as np
import pandas as pd
from deepifsac import DeepIFSACImputer, TabularPreprocessor

# Prepare data with NaN indicating missing values
# pandas DataFrame columns with object/category dtype are auto-detected as categorical
df = pd.read_csv("your_data.csv")

# Impute missing values
imputer = DeepIFSACImputer(
    pretrain=True,       # enable contrastive pretraining
    pretrain_epochs=100,
    embedding_size=32,
    device='auto',       # auto-select GPU/CPU
)
imputer.fit(df)
X_imputed = imputer.transform(df)  # np.ndarray, same shape as input

# Extract Transformer embeddings for downstream ML
X_embed = imputer.get_features(df)  # shape: (n_samples, embedding_size * n_features)

# Specify categorical columns explicitly for ndarray input
imputer2 = DeepIFSACImputer(cat_features=[0, 2], pretrain=False)
imputer2.fit(X_train)

# Use TabularPreprocessor standalone (compatible with sklearn Pipeline)
preprocessor = TabularPreprocessor()
preprocessor.fit(df)
processed = preprocessor.transform(df)
# processed keys: 'X_cat', 'X_con', 'cat_mask', 'con_mask', 'X_combined', 'nan_mask'
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pretrain` | `True` | Enable contrastive pretraining |
| `pretrain_epochs` | `100` | Number of pretraining epochs |
| `embedding_size` | `32` | Transformer embedding dimension |
| `transformer_depth` | `6` | Number of Transformer layers |
| `attention_heads` | `8` | Number of attention heads |
| `attention_type` | `'colrow'` | Attention type (`col`, `colrow`, `row`, etc.) |
| `missing_rate` | `0.3` | Artificial missing rate during training |
| `device` | `'auto'` | Device (`auto`, `cpu`, `cuda:0`) |
| `random_state` | `42` | Random seed for reproducibility |

---

## Project Structure

```
DeepIFSAC/
├── deepifsac/
│   ├── __init__.py          # Public package API
│   ├── augmentations.py
│   ├── corruptor.py
│   ├── data_openml.py
│   ├── pretraining.py
│   ├── imputer/
│   │   ├── __init__.py
│   │   ├── imputer.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── pretrainmodel.py
│   └── missingness/
│       ├── __init__.py
│       ├── sampler.py
│       └── utils.py
├── imputer/
│   ├── __init__.py          # Public API: DeepIFSACImputer, TabularPreprocessor
│   ├── imputer.py           # DeepIFSACImputer (sklearn-compatible)
│   └── preprocessor.py      # TabularPreprocessor (cat/con split, encoding, masking)
├── models/
│   ├── pretrainmodel.py     # DeepIFSAC model definition
│   └── model.py             # Transformer, MLP building blocks
├── data_openml/
│   └── my_data_prep_openml.py  # OpenML data loading and preprocessing
├── augmentations.py         # Data augmentation (CutMix, MixUp, embed_data_mask)
├── pretraining.py           # DeepIFSAC_pretrain() function
├── corruptor.py             # Corruption strategies (draw, noise, KNN, MICE)
├── my_train.py              # Main training script (OpenML pipeline)
├── pyproject.toml           # uv environment definition
├── difsac_env.yml           # conda environment definition
└── tests/
    └── imputer/
        ├── test_preprocessor.py
        └── test_imputer.py
```
