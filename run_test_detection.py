#!/usr/bin/env python3
"""
Run D-FINE detection on chicken-detection-labelme-format/coco-format/test images
using the latest checkpoint for the selected variant and draw results with supervision. Saves to test-result.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import supervision as sv

import configs

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
# D-FINE must be on path for src.core imports
DFINE_ROOT = PROJECT_ROOT / "D-FINE"
sys.path.insert(0, str(DFINE_ROOT))

from src.core import YAMLConfig

TEST_IMAGE_DIR = PROJECT_ROOT / "chicken-detection-labelme-format/coco-format/test"
OUTPUT_DIR = PROJECT_ROOT / "chicken-detection-labelme-format/coco-format/test-result"

CLASS_NAMES = ["chicken", "not-chicken"]
CONFIDENCE_THRESHOLD = 0.25
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VARIANT_CHOICES = ["n", "s", "m", "l", "x"]


def get_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Prefer best_stg1.pth, else last.pth, else newest .pth."""
    best = checkpoint_dir / "best_stg1.pth"
    if best.exists():
        return best
    last = checkpoint_dir / "last.pth"
    if last.exists():
        return last
    pths = list(checkpoint_dir.glob("*.pth"))
    if not pths:
        raise FileNotFoundError(f"No .pth checkpoint found in {checkpoint_dir}")
    return max(pths, key=lambda p: p.stat().st_mtime)


def load_model(config_path: Path, checkpoint_path: Path, device: str = "cuda:0"):
    """Load D-FINE model from config and checkpoint. Returns (model, img_size)."""
    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    cfg.model.load_state_dict(state)

    img_size = tuple(cfg.yaml_cfg["eval_spatial_size"])  # (H, W) for Resize

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device)
    model.eval()
    return model, img_size


def run_inference(model, image_path: Path, device: str, img_size: tuple[int, int]):
    """Run detection on one image. Returns (labels, boxes_xyxy, scores) on CPU."""
    im_pil = Image.open(image_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(device)

    transforms = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        labels, boxes, scores = model(im_data, orig_size)

    # Batch size 1
    labels = labels[0].cpu().numpy()
    boxes = boxes[0].cpu().numpy()
    scores = scores[0].cpu().numpy()
    return im_pil, labels, boxes, scores


def to_supervision_detections(labels, boxes, scores, conf_thr: float):
    """Convert model output to supervision Detections (filter by confidence)."""
    mask = scores >= conf_thr
    if not np.any(mask):
        return sv.Detections.empty()

    xyxy = boxes[mask].astype(np.float32)
    class_id = labels[mask].astype(int)
    confidence = scores[mask].astype(np.float32)
    return sv.Detections(xyxy=xyxy, class_id=class_id, confidence=confidence)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run D-FINE on test images, draw with supervision")
    parser.add_argument("--variant", type=str, choices=VARIANT_CHOICES, default="s", help="Model variant (default: s)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0 or cpu)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (default: latest in model dir)")
    args = parser.parse_args()

    variant = args.variant
    config_path = PROJECT_ROOT / configs.CONFIG_PATHS[variant]
    checkpoint_dir = PROJECT_ROOT / configs.OUTPUT_DIRS[variant]

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    if not checkpoint_dir.exists():
        print(f"Checkpoint dir not found: {checkpoint_dir}")
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else get_latest_checkpoint(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    print(f"Using checkpoint: {checkpoint_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading D-FINE-{variant.upper()} model on {device}...")
    model, img_size = load_model(config_path, checkpoint_path, device)

    image_paths = [
        p for p in TEST_IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort()
    print(f"Found {len(image_paths)} images in {TEST_IMAGE_DIR}")

    # Chicken = green, not-chicken = blue (class_id 0 and 1)
    color_palette = sv.ColorPalette([
        sv.Color.from_hex("#00FF00"),  # chicken
        sv.Color.from_hex("#0000FF"),  # not-chicken
    ])
    box_annotator = sv.BoxAnnotator(thickness=2, color=color_palette)
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_LEFT, text_scale=0.5, color=color_palette
    )

    for i, image_path in enumerate(image_paths):
        im_pil, labels, boxes, scores = run_inference(model, image_path, device, img_size)
        detections = to_supervision_detections(labels, boxes, scores, args.conf)
        frame = np.array(im_pil)

        if detections.xyxy is not None and len(detections.xyxy) > 0:
            labels_sv = [
                f"{CLASS_NAMES[c]} {s:.2f}"
                for c, s in zip(detections.class_id, detections.confidence)
            ]
            frame = box_annotator.annotate(scene=frame, detections=detections)

        out_path = OUTPUT_DIR / image_path.name
        Image.fromarray(frame).save(out_path)
        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f"  Saved {i + 1}/{len(image_paths)} -> {out_path.name}")

    print(f"Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
