<!-- GITHUB REAMDE -->

# Label Swapping for Imbalanced Music Datasets

<!-- Linked to their profiles -->
**Authors:** [Matei Cananau](https://github.com/matdrox), [Gustaw Siedlarski](https://github.com/gustawsi), [Alexander Själander](https://github.com/Soultoo)  
**Institution:** KTH Royal Institute of Technology

## Abstract

Data imbalance is a pervasive issue in machine learning that often hinders classifier performance by biasing predictions towards the majority class. **Label-Noise-based Re-balancing (LNR)** is a new method proposed by Hu et al. (2025) that introduces beneficial label noise to adjust biased decision boundaries without the drawbacks of traditional resampling techniques.

This repository contains the implementation and reproduction of the LNR method, applied to:
1.  **Binary Classification:** Reproduction of results on 4 datasets from the KEEL repository.
2.  **Multiclass Classification:** Reproduction of results on the CIFAR-10 image dataset.
3.  **Novel Application:** Application of LNR to the imbalanced music dataset **Meter2800** for time signature detection, specifically aiming to improve the detection of rare, irregular meters (e.g., 5/4, 7/4) using a ResNet18 architecture.

Our results demonstrate that beneficial label noise yields improved performance for recognizing minority class time signatures, albeit with a trade-off regarding overall accuracy.

## Table of Contents
- [Methodology](#methodology)
  - [LNR Algorithm](#lnr-algorithm)
  - [Time Signature Detection (ResNet18)](#time-signature-detection-resnet18)
- [Datasets](#datasets)
- [Installation and Usage](#installation-and-usage)
- [Experiments and Results](#experiments-and-results)
- [References](#references)

## Methodology

### LNR Algorithm
The LNR algorithm addresses class imbalance by introducing asymmetric label noise. Instead of assuming the majority class labels are always correct, the method selectively flips the labels of "confusing" majority samples (those with high posterior probabilities of resembling the minority class) to the minority class.

This process is controlled by the hyperparameter $t_{flip}$ (threshold), which determines the strictness of the flipping condition based on the standardized Z-score of the sample's posterior probability.

### Time Signature Detection (ResNet18)
Following the methodology of Abimbola et al. (2024), we approach time signature detection as a supervised image classification problem. Raw audio signals are converted into **Mel-frequency Cepstral Coefficients (MFCCs)** with input dimensions fixed to $130 \times 13$ (time frames $\times$ frequency coefficients).

We utilize a **ResNet18** architecture adapted for single-channel audio features to classify musical meters into four classes: $\frac{3}{4}$, $\frac{4}{4}$ (Majority) and $\frac{5}{4}$, $\frac{7}{4}$ (Minority).

## Datasets

This project utilizes three distinct datasets:

1.  **KEEL:** A repository of binary imbalanced datasets. We utilize `abalone`, `ecoli1`, `glass0`, and `vehicle1`.
2.  **CIFAR-10:** A standard image classification benchmark used for multiclass reproduction.
3.  **Meter2800:** An imbalanced dataset of 2,800 audio clips (30 seconds each) labeled with time signatures. It exhibits severe imbalance, with 1,200 samples each for $\frac{3}{4}$ and $\frac{4}{4}$, but only 200 samples each for $\frac{5}{4}$ and $\frac{7}{4}$.