# HAPS

**HAPS: Horizon-Adaptive Predictable Subspace Learning for Long-Horizon Multivariate Forecasting**

## Repository structure

```text
HAPS/
├── data_provider/
├── exp/
│   └── exp_basic.py
│   └── exp_long_term_forecasting_haps.py
├── models/
│   └── haps.py
├── scripts/HAPS/
│   ├── traffic.sh
│   └── traffic_ablation_h720.sh
├── dataset/README.md
├── run_haps.py
├── ENVIRONMENT.md
├── CITATION.cff
├── NOTICE
└── LICENSE
```

## Installation

```bash
conda create -n haps python=3.11
conda activate haps
pip install -r requirements.txt
```

The experiments used PyTorch 2.6.0, CUDA 12.4, cuDNN 9.1.

## Dataset preparation

For example, place the dataset file at:

```text
dataset/traffic/traffic.csv
```

See `dataset/README.md` for the public source.

## Run Traffic forecasting

```bash
bash scripts/HAPS/traffic.sh
bash scripts/HAPS/traffic.sh 96
bash scripts/HAPS/traffic.sh 192 336 720
```

## Traffic ablation study

```bash
bash scripts/HAPS/traffic_ablation_h720.sh
bash scripts/HAPS/traffic_ablation_h720.sh no_subspace
bash scripts/HAPS/traffic_ablation_h720.sh fixed_rank generic_decoder
```

## Acknowledgment

This implementation is built on
[Time-Series-Library](https://github.com/thuml/Time-Series-Library). The upstream MIT license is retained.
