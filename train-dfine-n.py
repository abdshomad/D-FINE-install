#!/usr/bin/env python3
"""
Train D-FINE N model on chicken detection dataset.

Usage:
    python train-dfine-n.py [--use-amp] [--seed SEED] [--resume PATH] [--tune PATH] [--test-only]
"""

import argparse
import sys
from pathlib import Path

import configs
from train_dfine_common import run_training

VARIANT = "n"
DISPLAY_NAME = "N"


def main():
    parser = argparse.ArgumentParser(
        description="Train D-FINE-N model on chicken detection dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--use-amp", action="store_true", default=configs.DEFAULT_TRAINING_CONFIG["use_amp"], help="Use automatic mixed precision training")
    parser.add_argument("--seed", type=int, default=configs.DEFAULT_TRAINING_CONFIG["seed"], help="Random seed")
    parser.add_argument("-r", "--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("-t", "--tune", type=str, default=None, help="Fine-tune from checkpoint path")
    parser.add_argument("--test-only", action="store_true", help="Only run testing/evaluation")
    parser.add_argument("--dfine-path", type=str, default=None, help="Path to D-FINE repository (if not in standard location)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (overrides config)")
    args = parser.parse_args()

    dfine_path = Path(args.dfine_path).resolve() if args.dfine_path else None
    code = run_training(
        VARIANT,
        use_amp=args.use_amp,
        seed=args.seed,
        resume=args.resume,
        tune=args.tune,
        test_only=args.test_only,
        output_dir_override=args.output_dir,
        dfine_path=dfine_path,
        variant_display_name=DISPLAY_NAME,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
