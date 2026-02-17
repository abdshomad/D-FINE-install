# D-FINE-install

Training setup for [D-FINE](https://github.com/abdshomad/D-FINE) (Redefine Regression Task of DETRs as Fine-grained Distribution Refinement) on chicken detection dataset using COCO format annotations.

## Overview

This repository provides a streamlined setup for training D-FINE models on a custom chicken detection dataset. The dataset contains 2 classes: `chicken` and `not-chicken`, formatted in COCO annotation format. All five D-FINE variants are supported: **N, S, M, L, X** (all HGNetv2-based).

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU(s)
- Git submodules initialized

### Setup

1. **Initialize git submodules:**
   ```bash
   git submodule update --init --recursive
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   uv venv
   uv sync
   ```

3. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

### Dataset Structure

The dataset is located in `chicken-detection-labelme-format/coco-format/`:

```
coco-format/
├── train/
│   ├── _annotations.coco.json
│   └── [training images]
├── valid/
│   ├── _annotations.coco.json
│   └── [validation images]
└── test/
    ├── _annotations.coco.json
    └── [test images]
```

**Dataset Configuration:**
- **Training annotations:** `chicken-detection-labelme-format/coco-format/train/_annotations.coco.json`
- **Validation annotations:** `chicken-detection-labelme-format/coco-format/valid/_annotations.coco.json`
- **Number of classes:** 2 (chicken, not-chicken)
- **Category remapping:** Disabled (`remap_mscoco_category: False`)

### Backbone / Checkpoints

All D-FINE variants (N, S, M, L, X) use HGNetv2 backbones. Backbones download automatically during training; no manual setup required.

## Training

### Train All Variants

Train all five model variants (N → S → M → L → X) sequentially. Automatically uses all available GPUs via `torchrun`:

```bash
./train-dfine-all-variants.sh --use-amp

# Or directly with uv
uv run train-dfine-all-variants.py --use-amp
```

Options:
- `--variants n s m l x` — Train specific variants only (default: all)
- `--use-amp` — Enable mixed precision
- `--test-only` — Evaluation only

### Single Variant Training

Train any of the five variants:

```bash
uv run train-dfine-n.py --use-amp
uv run train-dfine-s.py --use-amp
uv run train-dfine-m.py --use-amp
uv run train-dfine-l.py --use-amp
uv run train-dfine-x.py --use-amp
```

### Multi-GPU Training

The **train-dfine-all-variants.py** script automatically detects and uses all available GPUs via `torchrun`. For single-variant multi-GPU training, use `torchrun` from the D-FINE directory (see D-FINE repo for exact usage).

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--use-amp` | Enable Automatic Mixed Precision training | False |
| `--seed SEED` | Random seed for reproducibility | 0 |
| `-r, --resume PATH` | Resume training from checkpoint (single-variant scripts) | None |
| `-t, --tune PATH` | Fine-tune from checkpoint (single-variant scripts) | None |
| `--test-only` | Only run evaluation/testing | False |
| `--variants n s ...` | Variants to train (all-variants script only) | All |
| `--dfine-path PATH` | Path to D-FINE repository | `D-FINE` |
| `--output-dir DIR` | Output directory (single-variant scripts) | From config |

### Examples

- Train all variants with AMP: `./train-dfine-all-variants.sh --use-amp`
- Train specific variants: `uv run train-dfine-all-variants.py --variants s m --use-amp`
- Train single variant: `uv run train-dfine-s.py --use-amp`
- Resume: `uv run train-dfine-m.py --resume chicken-detection-models/dfine-m/last.pth`
- Evaluation only: `uv run train-dfine-s.py --test-only -r chicken-detection-models/dfine-s/last.pth`

## Test / Inference

Run detection on test images and save visualizations:

```bash
uv run run_test_detection.py --variant s
```

Options: `--variant n|s|m|l|x` (default: `s`), `--device cuda:0`, `--conf 0.25`, `--checkpoint PATH`.

Results are saved to `chicken-detection-labelme-format/coco-format/test-result/`.

## Configuration

### Global Configuration (`configs.py`)

- **Dataset paths:** Training and validation annotation files
- **Number of classes:** 2 (chicken, not-chicken)
- **D-FINE paths:** Path to D-FINE submodule and `train.py`
- **Config file paths:** Chicken configs in `configs/dfine_chicken/` for all five variants
- **Output directories:** `chicken-detection-models/dfine-n`, `chicken-detection-models/dfine-s`, … `chicken-detection-models/dfine-x`

### Chicken Configs

Chicken-specific YAML configs live in **`configs/dfine_chicken/`** in this repo. Each file includes the corresponding D-FINE base config from the submodule and overrides dataset paths and `num_classes` for the chicken dataset.

## Project Structure

```
D-FINE-install/
├── configs.py                           # Global configuration
├── configs/dfine_chicken/               # Chicken dataset configs (5 variants)
├── configs/dataset/chicken_detection.yml # Chicken dataset paths and num_classes
├── train_dfine_common.py                # Shared training logic
├── train-dfine-all-variants.py          # Train all variants (auto multi-GPU)
├── train-dfine-all-variants.sh
├── train-dfine-n.py ... train-dfine-x.py  # Single-variant scripts
├── run_test_detection.py                # Run detection on test images
├── D-FINE/                              # D-FINE submodule
├── chicken-detection-labelme-format/    # Dataset submodule
│   └── coco-format/
└── pyproject.toml
```

## Dependencies

Managed via `uv` and `pyproject.toml`:

```bash
uv sync
```

## Contributing

This repository uses git submodules. Do not modify files inside **D-FINE/** or **chicken-detection-labelme-format/**; see `AGENTS.md`.

## References

- [D-FINE](https://github.com/abdshomad/D-FINE) — Redefine Regression Task of DETRs as Fine-grained Distribution Refinement [ICLR 2025 Spotlight]

## License

See the D-FINE repository for license information.
