# Shared logic for D-FINE training scripts. Do not run directly.
# Used by train-dfine-{variant}.py and train-dfine-all-variants.py

import os
import socket
import sys
import subprocess
from pathlib import Path

import configs


def _find_free_port(base: int = 29500, range_size: int = 100) -> int:
    """Return a free port in [base, base + range_size). Prefer base if free."""
    for offset in range(range_size):
        port = base + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    return base  # fallback, may still fail


def run_training(
    variant: str,
    *,
    use_amp: bool = False,
    seed: int = 0,
    resume: str | None = None,
    tune: str | None = None,
    test_only: bool = False,
    output_dir_override: str | None = None,
    dfine_path: Path | None = None,
    variant_display_name: str | None = None,
    nproc_per_node: int = 0,
    master_port: int | None = None,
) -> int:
    """Run D-FINE train.py for the given variant. Returns exit code."""
    if variant_display_name is None:
        variant_display_name = variant.upper()

    root = dfine_path if dfine_path is not None else configs.DFINE_PATH
    train_script = root / "train.py"

    if not train_script.exists():
        print(f"Error: D-FINE train.py not found at {train_script}")
        print("\nPlease ensure D-FINE submodule is initialized:")
        print("   git submodule update --init --recursive")
        return 1

    config_path = (configs.PROJECT_ROOT / configs.CONFIG_PATHS[variant]).resolve()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    # Script and its args (train.py -c config ...). For torchrun we pass this as the program to run.
    script_args = [str(train_script), "-c", str(config_path)]
    if use_amp:
        script_args.append("--use-amp")
    if seed is not None:
        script_args.extend(["--seed", str(seed)])
    if resume:
        script_args.extend(["-r", resume])
    if tune:
        script_args.extend(["-t", tune])
    if test_only:
        script_args.append("--test-only")
    out_dir = output_dir_override if output_dir_override else str(Path("..") / configs.OUTPUT_DIRS[variant])
    script_args.extend(["--update", f"output_dir={out_dir}"])

    if nproc_per_node and nproc_per_node > 0:
        port = master_port if master_port is not None else _find_free_port()
        # torchrun runs: script [script_args]. Do not pass sys.executable as the script.
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc_per_node}",
            f"--master_port={port}",
            *script_args,
        ]
        distributed_port = port
    else:
        cmd = [sys.executable] + script_args
        distributed_port = None

    print("\n" + "=" * 60)
    print(f"D-FINE-{variant_display_name} Training Configuration")
    print("=" * 60)
    print(f"Config file: {config_path}")
    print(f"Train annotations: {configs.DATASET_CONFIG['train_ann_file']}")
    print(f"Val annotations: {configs.DATASET_CONFIG['val_ann_file']}")
    print(f"Number of classes: {configs.DATASET_CONFIG['num_classes']}")
    print(f"Classes: {', '.join(configs.DATASET_CONFIG['class_names'])}")
    print(f"D-FINE train.py: {train_script}")
    if resume:
        print(f"Resume from: {resume}")
    if tune:
        print(f"Fine-tune from: {tune}")
    print("Mode: Testing/Evaluation only" if test_only else "Mode: Training")
    if use_amp:
        print("AMP: Enabled")
    print(f"Seed: {seed}")
    if distributed_port is not None:
        print(f"Distributed: torchrun nproc_per_node={nproc_per_node} master_port={distributed_port}")
    print("=" * 60 + "\n")

    original_cwd = os.getcwd()
    try:
        os.chdir(root)
        print(f"Running: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, check=False)
        return result.returncode
    finally:
        os.chdir(original_cwd)
