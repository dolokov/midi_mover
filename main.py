"""Project entry point for midi_mover."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from midi_mover.app import run  # noqa: E402
from midi_mover.highscore_storage import (  # noqa: E402
    HighscoreStorageError,
    initialize_default_highscore_storage,
)


if __name__ == "__main__":
    try:
        initialize_default_highscore_storage()
    except HighscoreStorageError as exc:
        print(f"Highscore storage initialization failed: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
    raise SystemExit(run())