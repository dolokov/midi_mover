"""Finetune a YOLO pose model for hand keypoints with structured checkpoint output."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import os 

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

    from ultralytics import YOLO, settings

    settings.update({"datasets_dir": str(datasets_root)})

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(checkpoints_root),
        name=run_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
