# Multi-Temporal Land-Use / Land-Cover Change Detection

**Landsat/Sentinel-ready • raster processing • change detection • transition matrices • spatial QA/QC**

This project demonstrates a reproducible workflow for comparing classified land-cover rasters across time and quantifying how landscapes change between acquisition dates. It is designed for applications such as urban growth, vegetation loss, agricultural expansion, water change, and environmental monitoring.

## Project goals

- harmonize classified rasters to a common CRS, extent, and resolution
- validate class codes and NoData handling before comparison
- calculate pixel-level land-cover transitions between two dates
- generate a transition matrix and class-by-class change summary
- map persistent classes, gains, losses, and conversions
- export machine-readable summaries for downstream analysis
- document assumptions and QA/QC checks for reproducibility

## Example workflow

```text
Classified raster T1
        |
        +--> CRS / resolution / extent QA
        |
Classified raster T2
        |
        +--> CRS / resolution / extent QA
                  |
                  v
            Raster alignment
                  |
                  v
         Pixel transition coding
                  |
        +---------+----------+
        |                    |
        v                    v
Transition matrix      Change raster
        |                    |
        v                    v
Area statistics       Spatial interpretation
```

## Methods demonstrated

- multi-temporal raster comparison
- raster alignment and resampling checks
- categorical transition coding
- transition matrices
- class persistence, gain, and loss calculations
- urban-growth and environmental-change interpretation
- spatial QA/QC and reproducible reporting

## Suggested data sources

The workflow is compatible with classified products derived from:

- Landsat Collection 2
- Sentinel-2 Level-2A
- NAIP or other aerial imagery
- manually interpreted reference maps
- existing land-cover products such as NLCD

## Reproducibility and data transparency

This repository section is structured as a reusable workflow. Any demonstration inputs should be clearly labeled as public or synthetic. Reported change statistics should only be interpreted as real-world findings when they are produced from documented measured datasets with appropriate validation.

## Relevance to geospatial ML production

This project demonstrates several skills used in production geospatial analytics:

- large raster handling
- preprocessing and QA/QC
- repeatable change-detection logic
- standardized outputs
- technical documentation
- preparation of analysis-ready products for downstream machine learning or decision-support systems

## Author

**Priyanka Belbase**  
Geospatial Data Science | Remote Sensing | GeoAI | Machine Learning
