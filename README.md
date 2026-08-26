# Geospatial Land-Cover ML

Two complementary, reproducible machine-learning workflows for vegetation and land-cover analysis.

## Featured: PyTorch Deep Learning for Weed-Species Classification

**PyTorch • ResNet-18 • transfer learning • vegetation classification • macro F1 • reproducible evaluation**

[Open the complete DeepWeeds PyTorch workflow →](deepweeds_pytorch/)

This extension uses the public DeepWeeds field-image dataset to demonstrate an end-to-end deep-learning workflow for multiclass weed/vegetation classification. It includes data acquisition, official train/validation/test folds, augmentation, transfer learning, checkpointing, evaluation, confusion matrices, inference, tests, and reproducibility controls.

> DeepWeeds results are not pre-claimed. Performance metrics are produced only after a reproducible training/evaluation run on the downloaded public dataset.

## Spatial Validation Benchmark

The original project demonstrates **why spatial validation matters** for geospatial land-cover classification.

> **Transparency:** committed data and metrics in this section are synthetic demonstration outputs, not production mapping results.

### Technical highlights

- Multispectral-style features: blue, red, NIR, SWIR, NDVI
- Random Forest classification
- Random train/test split vs spatial holdout benchmark
- Feature importance and machine-readable results
- Runnable script, notebook, tests, figures, and sample data

### Results

| Validation | Accuracy | Macro F1 |
|---|---:|---:|
| Random split | **0.980** | **0.982** |
| Spatial holdout | **0.420** | **0.151** |

![Training map](figures/training_map.svg)
![Validation comparison](figures/validation_comparison.svg)
![Feature importance](figures/feature_importance.svg)

### Run the spatial-validation demo

```bash
pip install -r requirements.txt
python scripts/run_demo.py
python -m pytest -q
```

## Why the two workflows belong together

The spatial-validation benchmark focuses on **defensible geospatial evaluation**, while the DeepWeeds extension focuses on **modern computer vision with PyTorch**. Together they demonstrate model development and validation skills relevant to land-cover mapping, vegetation classification, agricultural remote sensing, habitat modeling, invasive-species surveillance, and environmental prediction.
