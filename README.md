# Physics-Informed CNN-Mamba for sEMG-Based Continuous Estimation

This repository contains the supplementary source code and a validation dataset for the manuscript: **"Physics-Informed CNN-Mamba for sEMG-Based Continuous Estimation of Lumbar Kinematics and Equivalent Torque"** submitted to *Biomedical Signal Processing and Control*.

## 1. Data Availability & Privacy Policy
To comply with data privacy policies, ethical restrictions (human subjects), and file size limits, this repository provides only a representative subset of the data used in the study. 

The full experimental dataset (all 8 subjects, 14 conditions, 3 trials per condition) is not publicly available. Researchers who wish to access the complete dataset for validation or replication purposes should contact the corresponding author via email.

## 2. Repository Structure
The repository is organized into two main directories: `data/` and `code/`. *(如果你本地文件夹已经改名叫 data 了，这里就写 data/；如果还没改，就照实写你那个长名字)*

### 2.1 Validation Dataset
*   `tree.txt`: A text file illustrating the complete hierarchical directory structure of the 8 anonymized subjects (Subject1 - Subject8) used in our full experiment.
*   `1.csv`, `2.csv`, `3.csv`: Three representative continuous trial datasets. Each CSV contains multi-channel raw sEMG signals (R-IL, L-IL, L-LO, R-LO) and motion capture kinematics (C7 and L5 spatial coordinates).

### 2.2 Source Code
Contains the complete Python pipeline, from data preprocessing to model evaluation and plotting.
*   `dataset.py`: Implements the `VRDLRDataset` class for data loading, Butterworth bandpass filtering, RMS feature extraction, and physics-label derivation.
*   `mamba_layer.py`: The core Selective State-Space Model architecture with customized dt/A initialization to ensure gradient stability.
*   `model.py`: Defines the complete `LumbarCNNMamba` network bridging the 1D-CNN backbone and Mamba sequence modeling.
*   `pretrain.py`: Script for generalized base model training across subjects.
*   `transfer_train.py`: Script for personalized fine-tuning using the proposed output-space physical consistency loss.
*   `speed_test.py`: Evaluates the single forward-inference latency to validate real-time edge deployment viability.
*   `evaluate.py` & `evaluate_and_plot.py`: Comprehensive scripts to calculate quantitative metrics (RMSE, MAE, R2) and generate publication-ready high-resolution figures.

## 3. System Requirements
*   **OS:** Windows Subsystem for Linux 2 (WSL2) - Ubuntu
*   **Python:** 3.8
*   **CUDA Toolkit:** 11.8
*   **Hardware Acceleration:** NVIDIA RTX GPU (Tested on RTX 4060)

## 4. Getting Started
We strongly recommend setting up a virtual environment (e.g., via Conda) before installation.
```bash
pip install -r requirements.txt

## 5. Contact
For any questions regarding the code, please open an issue in this repository. 
To request the full experimental dataset, please contact the corresponding author. The contact email is provided in the manuscript.
