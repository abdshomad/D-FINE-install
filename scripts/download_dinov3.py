#!/usr/bin/env python3
"""
Download DINOv3 models listed in secrets/DINOv3-download.md into pretrain/.
Creates pretrain/dinov3_vitb16_pretrain_lvd1689m.pth for config compatibility.
"""

import re
import shutil
import sys
from pathlib import Path

try:
    from urllib.request import urlretrieve
except ImportError:
    urlretrieve = None

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRETS_FILE = PROJECT_ROOT / "secrets" / "DINOv3-download.md"
OUT_DIR = PROJECT_ROOT / "pretrain"

# Config in chicken-detection-labelme-format expects this name (no hash)
VITB16_ALIAS = "dinov3_vitb16_pretrain_lvd1689m.pth"


def parse_markdown(path: Path) -> list[tuple[str, str]]:
    """Extract (filename, url) pairs from DINOv3-download.md."""
    text = path.read_text()
    lines = [ln.strip() for ln in text.splitlines()]
    pairs = []
    for i, line in enumerate(lines):
        if not line.startswith("https://"):
            continue
        # Previous non-empty line should be the .pth filename
        for j in range(i - 1, -1, -1):
            prev = lines[j]
            if not prev:
                continue
            if re.match(r"^dinov3_.*\.pth$", prev):
                pairs.append((prev, line))
            break
    return pairs


def download_file(url: str, dest: Path) -> bool:
    """Download url to dest. Return True on success."""
    if urlretrieve is None:
        print("urllib.request.urlretrieve not available", file=sys.stderr)
        return False
    try:
        urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    if not SECRETS_FILE.exists():
        print(f"Secrets file not found: {SECRETS_FILE}", file=sys.stderr)
        return 1

    pairs = parse_markdown(SECRETS_FILE)
    if not pairs:
        print("No (filename, url) pairs found in secrets file.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(pairs)} DINOv3 files to {OUT_DIR}")

    vitb16_full_name = None
    for name, url in pairs:
        dest = OUT_DIR / name
        if dest.exists():
            print(f"  Skip (exists): {name}")
            if name.startswith("dinov3_vitb16_pretrain_lvd1689m-"):
                vitb16_full_name = name
            continue
        print(f"  Downloading: {name}")
        if not download_file(url, dest):
            return 1
        if name.startswith("dinov3_vitb16_pretrain_lvd1689m-"):
            vitb16_full_name = name

    # Create alias for rtv4 config: pretrain/dinov3_vitb16_pretrain_lvd1689m.pth
    alias_path = OUT_DIR / VITB16_ALIAS
    if vitb16_full_name and not alias_path.exists():
        src = OUT_DIR / vitb16_full_name
        if src.exists():
            shutil.copy2(src, alias_path)
            print(f"  Created alias: {VITB16_ALIAS} -> {vitb16_full_name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
