# Historical Artifact Damage Segmentation

This repository contains the implementation of a semantic segmentation pipeline designed to detect and classify multi-class physical damages (e.g., cracks, peels, stains, folds) across historical art and artifact surfaces.

Built with **PyTorch**, this project evaluates **DeepLabV3+** architectures with **ResNet-50** and **ResNet-101** backbones using a **Leave-One-Material-Out Cross-Validation (LOMO-CV)** framework to ensure cross-substrate generalization.

## Key Features

- **Architecture:** Semantic segmentation using DeepLabV3+ with ResNet-50 and ResNet-101 feature extractors[cite: 2].
- **Robust Evaluation:** Evaluated via Leave-One-Material-Out Cross-Validation (LOMO-CV) to benchmark generalization across unseen art substrates (e.g., paper, canvas, wood)[cite: 2].
- **Pipeline Engineering:** Custom RGB-to-mask color mapping pre-processing, data augmentation, and combined Cross-Entropy loss functions[cite: 2].
- **Performance Optimization:** Automated Mixed-Precision training (`torch.cuda.amp.autocast` / `GradScaler`) and dynamic learning rate scheduling (`ReduceLROnPlateau`) for efficient memory usage and faster convergence[cite: 2].

## Tech Stack

- **Framework:** PyTorch, torchvision
- **Computer Vision:** OpenCV, PIL, albumentations
- **Data & Plotting:** NumPy, Matplotlib, pandas
