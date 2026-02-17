# Installation Guide

Complete step-by-step installation guide for D-FINE training setup on chicken detection dataset.

## Prerequisites

Before starting, ensure you have:

- **Python 3.11+** installed
- **CUDA-capable GPU(s)** with NVIDIA drivers installed
- **Git** installed
- **`uv`** package manager installed ([Installation guide](https://github.com/astral-sh/uv))
- **nvidia-smi** available (for GPU monitoring)

### Verify Prerequisites

```bash
python3 --version   # Should be 3.11 or higher
nvidia-smi          # Should show GPU information
uv --version
```

## Installation Steps

### Step 1: Clone Repository and Initialize Submodules

```bash
git clone <repository-url>
cd D-FINE-install

git submodule update --init --recursive
```

The repository uses git submodules for:
- **`D-FINE/`** — The D-FINE framework
- **`chicken-detection-labelme-format/`** — The dataset (COCO format)

### Step 2: Set Up Python Virtual Environment

```bash
uv venv
uv sync
```

Optional activation:
```bash
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
```

### Step 3: Backbone Weights

All D-FINE variants (N, S, M, L, X) use HGNetv2 backbones. Backbones download automatically during training; no manual step required.

### Step 3b: DINOv3 Weights (optional, for RT-DETRv4 in chicken-detection-labelme-format)

If you use RT-DETRv4 with DINOv3 teacher in `chicken-detection-labelme-format`, download the required DINOv3 models using the URLs in `secrets/DINOv3-download.md`:

```bash
uv run python scripts/download_dinov3.py
```

This downloads all listed DINOv3 models into **`pretrain/`** at the project root and creates `pretrain/dinov3_vitb16_pretrain_lvd1689m.pth` for config compatibility. Run training from the project root so `pretrain/` is found, or set `teacher_model.dinov3_weights_path` in the rtv4 config to `../../pretrain/dinov3_vitb16_pretrain_lvd1689m.pth` if running from another directory.

### Step 4: Verify Configuration

```bash
# Check D-FINE and dataset submodules exist
ls D-FINE/train.py
ls chicken-detection-labelme-format/coco-format/train/_annotations.coco.json

# Check chicken configs exist (in this repo)
ls configs/dfine_chicken/dfine_*.yml

# Check training scripts
ls train-dfine-*.py
```

### Step 5: Verify Dataset Structure

Ensure the dataset layout is:

```
chicken-detection-labelme-format/
└── coco-format/
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

## Configuration

### Global Configuration (`configs.py`)

- Dataset paths (training and validation annotations)
- Number of classes: 2 (chicken, not-chicken)
- D-FINE path and train script
- Config paths for all five variants in `configs/dfine_chicken/`
- Output directories: `chicken-detection-models/dfine-n`, … `chicken-detection-models/dfine-x`

### Chicken Configs

Chicken-specific YAMLs are in **`configs/dfine_chicken/`** in this repo. They include the corresponding D-FINE base config from the submodule and override dataset paths and `num_classes`.

## Testing Installation

### Quick Test

```bash
uv run python -c "import torch; import torchvision; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Test Training Script

```bash
# Should print configuration (may fail at resume if no checkpoint yet)
uv run python train-dfine-s.py --test-only -r /nonexistent 2>&1 | head -25
```

## First Training Run

### Start Training

```bash
# Train one variant (e.g. S) with AMP
uv run train-dfine-s.py --use-amp

# Or train all variants (uses all GPUs via torchrun)
./train-dfine-all-variants.sh --use-amp
```

## Troubleshooting

### Git Submodules Not Initialized

**Error:** `D-FINE/train.py not found`

**Solution:**
```bash
git submodule update --init --recursive
```

### CUDA Out of Memory

- Reduce batch size in the D-FINE base config (or override in chicken config if supported).
- Use a smaller variant (e.g. N or S).
- Free GPU memory before training (e.g. close other processes).

### Missing Python Dependencies

```bash
uv sync
# or
uv add <package-name>
```

### Path or Config Errors

- Chicken configs use paths relative to **D-FINE** (working directory when training): `../chicken-detection-labelme-format/coco-format/...`
- Ensure annotations exist at `chicken-detection-labelme-format/coco-format/train/_annotations.coco.json` and `valid/_annotations.coco.json`.

## Installation Checklist

- [ ] Python 3.11+ installed
- [ ] CUDA and GPU drivers installed (`nvidia-smi` works)
- [ ] `uv` installed
- [ ] Repository cloned
- [ ] Git submodules initialized (`D-FINE/` and `chicken-detection-labelme-format/`)
- [ ] Virtual environment created (`uv venv`) and deps installed (`uv sync`)
- [ ] Dataset and annotations present
- [ ] Chicken configs present in `configs/dfine_chicken/`
- [ ] Training scripts present (`train-dfine-*.py`)

## Next Steps

1. Review `configs.py` and `configs/dfine_chicken/`.
2. Start training: e.g. `uv run train-dfine-s.py --use-amp`.
3. Run inference: `uv run run_test_detection.py --variant s`.

See **README.md** for usage and **AGENTS.md** for development guidelines.

## License

See the D-FINE repository for license information.
