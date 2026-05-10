# TariqCViT — Research Scripts

Run these after training to generate paper-quality metrics.

## Setup

```bash
pip install scikit-learn scipy
```

## Scripts

| Script | Purpose | Time |
|--------|---------|------|
| `01_metrics.py` | Precision, Recall, F1, AUC-ROC, Cohen Kappa per class | ~2 min |
| `02_baselines.py` | Train ResNet50, VGG16, EfficientNet for comparison | ~2 hrs |
| `03_significance.py` | McNemar's test — proves CNN-ViT beats CNN significantly | ~3 min |
| `04_gradcam_figures.py` | Grad-CAM overlays for paper figures | ~5 min |

## Config

Edit the paths at the top of each script:

```python
CNN_MODEL    = 'models/cnn_best.keras'
HYBRID_MODEL = 'models/hybrid_best.keras'
TEST_DIR     = 'data/test'     # folder with one subfolder per class
TRAIN_DIR    = 'data/train'    # for baselines only
```

## Expected dataset structure

```
data/
  train/
    Mosaic Disease/
    Insect Infestation/
  val/
    Mosaic Disease/
    Insect Infestation/
  test/
    Mosaic Disease/
    Insect Infestation/
```

## Key outputs for your paper

- **01_metrics.py** → Table 3 (Classification report)
- **02_baselines.py** → Table 4 (Comparison with baselines)
- **03_significance.py** → Section 4.3 (Statistical analysis)
- **04_gradcam_figures.py** → Figure 5 (Grad-CAM visualisations)
