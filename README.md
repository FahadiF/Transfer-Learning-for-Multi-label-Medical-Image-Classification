# Transfer Learning for Multi-label Medical Image Classification

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

A deep learning project applying **transfer learning** to tackle multi-label classification of medical images, addressing the challenge of limited labeled data in clinical settings.

---

## Problem Statement

Obtaining large-scale labeled medical datasets is difficult due to:
- Patient privacy regulations
- High expert annotation costs
- Limited availability of specialist knowledge

This project addresses these constraints by leveraging **pretrained backbone models** to boost performance on small-scale medical datasets.

---

## Approach

- Fine-tune pretrained CNN backbones (stored in `pretrained_backbone/`) on a medical image dataset
- Apply multi-label classification (one image can have multiple pathology labels)
- Evaluate on both on-site and off-site test sets
- Compare custom implementations (see `Fahad/` and `Shane/` folders) against a shared baseline

---

## Repository Structure

```
.
├── Fahad/                  # Fahad's model implementation
├── Shane/                  # Shane's model implementation  
├── Report/                 # Project report
├── pretrained_backbone/    # Pretrained model weights
├── images/                 # Dataset images
├── Final_Code.ipynb        # Main training & evaluation notebook
├── code_template.py        # Base code template
├── train.csv               # Training labels
├── val.csv                 # Validation labels
├── onsite_test_submission.csv
└── offsite_test.csv
```

---

## Getting Started

```bash
git clone https://github.com/FahadiF/Transfer-Learning-for-Multi-label-Medical-Image-Classification.git
cd Transfer-Learning-for-Multi-label-Medical-Image-Classification
pip install torch torchvision jupyter pandas numpy
jupyter notebook Final_Code.ipynb
```

---

## Key Concepts

| Concept | Description |
|---|---|
| Transfer Learning | Reusing weights from models pretrained on ImageNet |
| Multi-label Classification | Each sample may belong to multiple classes simultaneously |
| Backbone Fine-tuning | Unfreezing top layers of pretrained CNN for task-specific training |
| Class Imbalance Handling | Weighted loss functions to handle skewed label distributions |

---

## Authors

- **Fahad Ibne Fahian** — [FahadiF](https://github.com/FahadiF)
- Shane

---

*University project — Applied Deep Learning*
