#!/usr/bin/env python3
"""
Train all D-FINE variants (N, S, M, L, X) on chicken detection dataset.

Runs training sequentially for each variant. Uses torchrun for multi-GPU when available.

Usage:
    python train-dfine-all-variants.py [--use-amp] [--seed SEED] [--test-only] [--variants n s ...]
"""

import argparse
import sys
from pathlib import Path

import torch

import configs
from train_dfine_common import run_training

VARIANT_NAMES = {
    "n": "N",
    "s": "S",
    "m": "M",
    "l": "L",
    "x": "X",
}


def main():
    parser = argparse.ArgumentParser(
        description="Train all D-FINE variants on chicken detection dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--use-amp", action="store_true", default=configs.DEFAULT_TRAINING_CONFIG["use_amp"], help="Use automatic mixed precision training")
    parser.add_argument("--seed", type=int, default=configs.DEFAULT_TRAINING_CONFIG["seed"], help="Random seed")
    parser.add_argument("--test-only", action="store_true", help="Only run testing/evaluation")
    parser.add_argument("--dfine-path", type=str, default=None, help="Path to D-FINE repository")
    parser.add_argument("--master-port", type=int, default=None, help="Master port for torchrun (default: auto-detect a free port)")
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        choices=configs.VARIANTS_ORDER,
        default=configs.VARIANTS_ORDER,
        help="Variants to train (default: all)",
    )
    args = parser.parse_args()

    dfine_path = Path(args.dfine_path).resolve() if args.dfine_path else None
    train_script = (dfine_path or configs.DFINE_PATH) / "train.py"
    if not train_script.exists():
        print(f"Error: D-FINE train.py not found at {train_script}")
        print("   git submodule update --init --recursive")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    nproc = num_gpus if num_gpus >= 1 else 0
    master_port = args.master_port

    print("\n" + "=" * 60)
    print("D-FINE All Variants Training")
    print("=" * 60)
    print(f"Variants: {[VARIANT_NAMES[v] for v in args.variants]}")
    print(f"D-FINE train.py: {train_script}")
    print(f"GPUs: {num_gpus} available, using {'torchrun (distributed)' if nproc else 'single process'}")
    print(f"AMP: {'Enabled' if args.use_amp else 'Disabled'}")
    print(f"Seed: {args.seed}")
    print(f"Mode: {'Testing only' if args.test_only else 'Training'}")
    print("=" * 60 + "\n")

    results = []
    for variant in args.variants:
        print("\n" + "-" * 60)
        print(f"Training D-FINE-{VARIANT_NAMES[variant]} ({variant})")
        print("-" * 60)
        code = run_training(
            variant,
            use_amp=args.use_amp,
            seed=args.seed,
            test_only=args.test_only,
            dfine_path=dfine_path,
            variant_display_name=VARIANT_NAMES[variant],
            nproc_per_node=nproc,
            master_port=master_port,
        )
        results.append((variant, code))
        if code != 0:
            print(f"\nWarning: D-FINE-{VARIANT_NAMES[variant]} exited with code {code}")

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    for variant, code in results:
        status = "OK" if code == 0 else f"FAILED ({code})"
        print(f"  D-FINE-{VARIANT_NAMES[variant]}: {status}")
    print("=" * 60 + "\n")

    exit_code = next((c for _, c in results if c != 0), 0)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
