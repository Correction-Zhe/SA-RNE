# SA-RNE

Structure-Aware Residual Noise Estimation for Lightweight Infrared Image Denoising.

SA-RNE is a lightweight PyTorch framework for clean-reference-free infrared image denoising. It targets residual fixed-pattern noise (FPN) and temporal fluctuations that remain after non-uniformity correction (NUC), especially when local contrast enhancement amplifies these weak residual artifacts into sparse, discontinuous, and locally fragmented patterns.

## Overview

Instead of assuming globally continuous stripe noise, SA-RNE estimates enhancement-amplified residual noise directly in the enhanced image domain. The inference model is a compact Conv+ReLU residual estimator:

```text
denoised = enhanced_input - estimated_residual_noise
```

During training, a Local Correlation Prior (LCP) and Bilateral Region Contrast (BRC) provide self-supervised structure-aware guidance from the noisy enhanced image itself. The training-only objective combines LCP consistency, gradient preservation, and flat-region total variation to suppress residual artifacts while preserving thermal structure, without adding inference-time overhead.

## Highlights

- Lightweight residual noise estimator with about 0.012M parameters under the default compact setting.
- No clean reference images are required for training.
- Structure-aware self-supervision with LCP and BRC.
- Training-only structure guidance; inference uses only the residual estimator.
- Designed for fragmented residual FPN after NUC and local contrast enhancement.
- Includes a synthetic fragmented stripe-noise generator for controlled experiments.

## Project Structure

```text
.
|-- config.py                  # Model, loss, training, and path configuration
|-- dataset.py                 # Infrared grayscale image dataset
|-- lcp_model.py               # Local Correlation Prior module
|-- loss.py                    # SA-RNE training objective
|-- main_model.py              # Compact residual noise estimator
|-- synthetic_stripe_noise.py  # Synthetic fragmented residual noise generator
`-- train.py                   # Training entry point
```

## Installation

Python 3.8+ is recommended.

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch instructions.

## Dataset

Place enhanced infrared images under:

```text
datasets/train/
```

Supported image formats are `png`, `jpg`, `jpeg`, `bmp`, `tif`, and `tiff`. Images are converted to grayscale during loading.

You can edit paths in `config.py`:

```python
dataset_dir = './datasets/train'
outputs_dir = './outputs'
```

## Training

```bash
python train.py
```

Outputs are written to `outputs/` by default:

```text
outputs/
|-- checkpoints/
|-- patches/
`-- visuals/
```

Checkpoints contain the residual estimator, training-only LCP module, optimizer, and scheduler states. For deployment, only the residual estimator is needed.

## Synthetic Residual Noise

Edit the input and output paths at the top of `synthetic_stripe_noise.py`, then run:

```bash
python synthetic_stripe_noise.py
```

The script simulates sparse, discontinuous residual stripe artifacts with edge-aware suppression, matching the fragmented residual noise setting discussed in the paper.

## Citation

If this repository is useful for your research, please cite the paper:

```bibtex
@article{sar_ne,
  title={SA-RNE: Structure-Aware Residual Noise Estimation for Lightweight Infrared Image Denoising},
  author={},
  journal={},
  year={}
}
```

## Notes

- The default paths are relative so the project can be cloned and run without machine-specific edits.
- Large datasets, generated outputs, and checkpoints should stay outside version control.
- This repository does not include pretrained weights or sample datasets.
