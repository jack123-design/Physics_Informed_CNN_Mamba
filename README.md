# Physics_Informed_CNN_Mamba
This repository contains the supplementary source code and a sample dataset for the manuscript: **"Physics-Informed CNN-Mamba for sEMG-Based Continuous Estimation of Lumbar Kinematics and Equivalent Torque"** submitted to *Biomedical Signal Processing and Control*.

 System Requirements
The code has been developed and thoroughly tested in the following environment:
OS: Windows Subsystem for Linux 2 (WSL2) - Ubuntu 
Python: 3.8
CUDA Toolkit:** 11.8 
Hardware Acceleration: NVIDIA RTX GPU (Tested on RTX 4060)

 Repository Structure
The supplementary material is organized into two main directories: `data/` and `code/`.
1. `data/`
To comply with data privacy policies and file size limits, this directory contains a representative subset of the data used in the study:
tree.txt`: A text file illustrating the complete hierarchical directory structure of the 8 anonymized subjects (Subject1 - Subject8) used in our full experiment.
sample.csv`: A sample continuous trial dataset containing multi-channel raw sEMG signals (R-IL, L-IL, L-LO, R-LO) and motion capture kinematics (C7 and L5 spatial coordinates).

 2. `code/`
Contains the complete Python pipeline, from data preprocessing to model evaluation and plotting.
dataset.py`: Implements the `VRDLRDataset` class for data loading, Butterworth bandpass filtering, RMS feature extraction, and physics-label derivation.
mamba_layer.py`: The core Selective State-Space Model architecture with customized dt/A initialization to ensure gradient stability.
model.py`: Defines the complete `LumbarCNNMamba` network bridging the 1D-CNN backbone and Mamba sequence modeling.
pretrain.py`: Script for generalized base model training across subjects.
transfer_train.py`: Script for personalized fine-tuning using the proposed output-space physical consistency loss.
speed_test.py`: Evaluates the single forward-inference latency to validate real-time edge deployment viability.
evaluate.py` & `evaluate_and_plot.py`: Comprehensive scripts to calculate quantitative metrics (RMSE, MAE, $R^2$) and generate publication-ready high-resolution figures.

 Getting Started

1. Install Dependencies**
We strongly recommend setting up a virtual environment (e.g., via Conda) before installation.
bash
pip install -r requirements.txt
