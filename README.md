# Geospatial Land-Cover Classification with Spatial Machine Learning

A Random Forest land-cover workflow that emphasizes a major issue in geospatial AI: **spatial data leakage**.

## What makes this project stronger than a basic ML notebook
Random train/test splitting can overestimate performance when neighboring samples are spatially autocorrelated. This repository uses spatial blocks to create a more defensible holdout set.

## Skills demonstrated
- Multispectral feature engineering
- NDVI and NDWI
- Random Forest classification
- Spatial block train/test splitting
- Feature importance
- Confusion-matrix evaluation
- Reproducible Python workflow

## Run
```bash
pip install -r requirements.txt
python src/demo.py
pytest -q
```

## Data note
The demonstration script generates synthetic land-cover samples. Replace them with field samples, labeled polygons, or satellite-extracted training points for a real project.

## Why recruiters should care
This project shows both machine-learning implementation and awareness of geospatial validation, which is essential for credible Earth-observation models.

## Author
Priyanka Belbase | GIS | Geospatial AI | Earth Observation | Machine Learning
