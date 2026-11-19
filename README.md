# Mediscan AI: Automated Brain Tumor Analysis
## A Two-Stage Deep Learning Pipeline for MRI Segmentation and Classification

## 📌 Project Overview
Mediscan AI is an intelligent medical imaging system developed to assist in the early diagnosis of brain tumors. This project implements a sophisticated two-stage pipeline: a U-Net architecture for pixel-level segmentation to localize the tumor, followed by a CNN to classify the detected tumor into specific categories.
By isolating the Region of Interest (ROI) from the segmented mask, the system effectively reduces background noise and focuses the classification model on the most relevant diagnostic features.

## 🚀 Key Features
**Standardized Preprocessing:** Full pipeline including grayscale conversion, intensity normalization (0 to 1), resizing to 256 × 256, and noise reduction via Gaussian filtering.
**Pixel-Level Segmentation:** Utilizes a U-Net architecture with symmetric encoder-decoder structures and skip connections to preserve fine spatial details.
**Automated ROI Extraction:** Isolates tumor regions from original MRI slices using predicted segmentation masks to enhance classification accuracy.
**Multi-Class Classification:** Categorizes tumors into three primary types: Glioma, Meningioma, and Pituitary Tumor.
**Comprehensive Evaluation:** Independent performance monitoring for segmentation (using Dice Coefficient and IoU) and classification (using Accuracy and Confusion Matrices).

## 🧠 Model Architecture & Methodology
![Mediscan AI Workflow](images/workflow_diagram.png)
The project validates an end-to-end framework through a modular strategy:
**Segmentation (U-Net):** Extracts high-level semantic information while maintaining boundary precision.
**ROI Generation:** Compares classification performance using full images vs. ground-truth ROIs vs. predicted ROIs to analyze how segmentation quality affects diagnostic results.
**Classification (CNN):** A high-performance classifier trained on localized tumor features.

## 📂 Dataset
The project uses the **Figshare brain tumor dataset** by Jun Cheng: 3,064 T1-weighted contrast-enhanced slices from 233 patients (meningioma 708, glioma 1,426, pituitary 930), with tumor masks and patient IDs.
- **Source:** https://doi.org/10.6084/m9.figshare.1512427 (licence CC BY 4.0).
- **Please cite:** J. Cheng et al., "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition," *PLoS ONE* 10(10): e0140381, 2015, https://doi.org/10.1371/journal.pone.0140381.
- **Not included in this repository** (size). Download the archive and extract it so that `dataset/dataset/data/1.mat … 3064.mat` and `dataset/dataset/cvind.mat` exist.
- **Changes we make:** the semester-2 pipeline converts the `.mat` files to 256×256 arrays (area resampling for images, nearest-neighbour for masks) and uses the dataset's own patient-disjoint 5-fold split (`cvind.mat`).

## ⚙️ Setup (semester 2)
Requires Python 3.11+ and, for training, an NVIDIA GPU (developed on an RTX 3050 Laptop, 4 GB).
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e . --no-deps
pytest                             # data/GPU tests are skipped when the data or GPU is missing
```
Semester-1 scripts are frozen in [`legacy/semester1/`](legacy/semester1/README.md).

## Results (Pilot Study)
### Visual Validation: The U-Net model demonstrated successful pixel-level localization, effectively distinguishing tumor tissue from healthy brain structures.
![Segmentation Comparison](images/segmentation_results1.png)

![Segmentation Comparison](images/segmentation_results2.png)

## 🗺️ Project Roadmap (2-Semester Research)
### Phase 1: Foundation (Semester 1 - Completed)
- Established a functional Phase 1 system with a pilot dataset.
- Implemented core U-Net and CNN architectures.
- Validated the feasibility of the segmentation-classification pipeline.

### Phase 2: Optimization (Semester 2 - In Progress)
- **Data Augmentation:** Implementing elastic deformations and intensity shifts to improve robustness.
- **Advanced Metrics:** Transitioning to Hausdorff Distance (HD95) for stricter boundary evaluation.
- **Architectural Upgrades:** Exploring Attention U-Net and Residual U-Net enhancements.
- **Backbone Optimization:** Replacing the SimpleCNN baseline with ResNet or EfficientNet.
