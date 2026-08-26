# Methodology

## Scientific question

Can a compact convolutional neural network distinguish multiple weed species from field RGB imagery under variable backgrounds and illumination?

## Dataset

This workflow uses **DeepWeeds**, an open dataset of 17,509 in-situ RGB images covering eight weed species and a negative class. The dataset was collected at multiple Queensland, Australia locations and is distributed under CC BY 4.0. The original paper reported 95.7% average classification accuracy using ResNet-50; that published benchmark is not a result of this repository.

## Experimental design

The workflow uses the official DeepWeeds train/validation/test CSV subsets. `fold=0` is the default, while folds 0–4 are supported for repeated experiments.

## Model

The default model is ImageNet-pretrained **ResNet-18** with the final classification layer replaced for nine DeepWeeds classes. ResNet-18 was selected as a transparent, computationally practical transfer-learning baseline rather than to reproduce the original paper exactly.

## Augmentation

Training augmentation includes random resized crops, horizontal flips, modest rotations, and color jitter. Validation/test preprocessing uses deterministic resize and center crop. ImageNet normalization is applied to all splits.

## Optimization

Training uses AdamW, cross-entropy loss, optional inverse-frequency class weighting, learning-rate reduction on validation macro F1, mixed precision on CUDA, and early stopping.

## Evaluation

The test workflow exports accuracy, macro F1, weighted F1, macro precision, macro recall, a class-level report, per-image predictions, and a confusion matrix. Macro F1 is emphasized because it treats each weed class equally despite class imbalance.

## Reproducibility

Random seeds are fixed, configuration is written to each run directory, official split files are used, and the code includes unit tests that do not require the large image archive.

## Limitations and transfer to ecological remote sensing

DeepWeeds uses ground RGB photographs rather than hyperspectral, UAV, or satellite imagery. Therefore, this project demonstrates **PyTorch/CNN image-classification workflow competence**, not direct invasive-species mapping at landscape scale. The same modeling structure can be extended to UAV image patches, multispectral tensors, or hyperspectral inputs with sensor-specific preprocessing and spatially independent validation.
