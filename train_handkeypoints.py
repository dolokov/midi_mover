"""Finetune a YOLO pose model for hand keypoints with structured checkpoint output."""

from __future__ import annotations

import argparse
from datetime import datetime
import os 
from pathlib import Path
import re


def _build_augmentation_overrides(mode: str) -> dict[str, float]:
    """Return Ultralytics augmentation overrides for the selected mode."""
    if mode == "heavy":
        # Stronger geometric + photometric augmentations to improve
        # orientation and illumination robustness for hand keypoints.
        return {
            "degrees": 35.0,
            "translate": 0.20,
            "scale": 0.50,
            "shear": 12.0,
            "perspective": 0.001,
            "flipud": 0.20,
            "fliplr": 0.50,
            "hsv_h": 0.03,
            "hsv_s": 0.90,
            "hsv_v": 0.70,
            "mosaic": 1.0,
            "mixup": 0.20,
        }

    # vanilla: preserve current behavior (Ultralytics defaults / existing setup)
    return {}

def _extract_variant(model_path: str) -> str:
    """Extract a compact variant tag from model filename, e.g. yolo26n-pose.pt -> 26n."""
    model_name = Path(model_path).name
    match = re.search(r"yolo([0-9]+[a-z])(?:-|\.)", model_name, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()

    stem = Path(model_path).stem.lower().replace("-pose", "")
    return stem or "custom"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/fine-tune a YOLO hand-keypoint model with organized checkpoints."
    )
    parser.add_argument("--model", default="yolo26n-pose.pt", help="Pretrained model path/name.")
    parser.add_argument(
        "--data",
        default= "/home/alex/data/midi_mover/handdatasets/hand-keypoints/data.yaml", #"hand-keypoints.yaml",
        help="Dataset YAML for keypoint training.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument(
        "--augmentation-mode",
        choices=("vanilla", "heavy"),
        default="vanilla",
        help="Augmentation preset. 'vanilla' keeps current/default behavior; 'heavy' increases geometric and illumination augmentation.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path(os.path.expanduser("~/data/midi_mover/handdatasets")),
        help="Directory where Ultralytics stores/downloads datasets.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    variant = _extract_variant(args.model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoints_root = Path(os.path.expanduser("~/data/midi_mover/handcheckpoints"))
    datasets_root = args.datasets_dir.expanduser().resolve()
    run_name = f"hand_{variant}_img{args.imgsz}_{timestamp}"

    checkpoints_root.mkdir(parents=True, exist_ok=True)
    datasets_root.mkdir(parents=True, exist_ok=True)

    print(f"[train_handkeypoints] Checkpoints root: {checkpoints_root}")
    print(f"[train_handkeypoints] Datasets root: {datasets_root}")
    print(f"[train_handkeypoints] Run name: {run_name}")
    print(
        "[train_handkeypoints] Full run dir: "
        f"{checkpoints_root / run_name}"
    )
    print(f"[train_handkeypoints] Augmentation mode: {args.augmentation_mode}")

    from ultralytics import YOLO, settings

    settings.update({"datasets_dir": str(datasets_root)})

    model = YOLO(args.model)
    augmentation_overrides = _build_augmentation_overrides(args.augmentation_mode)
    if augmentation_overrides:
        print(
            "[train_handkeypoints] Augmentation overrides: "
            f"{augmentation_overrides}"
        )
    else:
        print(
            "[train_handkeypoints] Augmentation overrides: none "
            "(using vanilla/current defaults)"
        )

    train_kwargs = {
        **augmentation_overrides,
    }

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(checkpoints_root),
        name=run_name,
        **train_kwargs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
