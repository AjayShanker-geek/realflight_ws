#!/usr/bin/env python3
"""
Wrapper for tools/preprocess_traj_new.py to generate 100 Hz trajectory CSVs.
"""

from pathlib import Path
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root / "tools"))
    try:
        from preprocess_traj_new import main as preprocess_main
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import preprocess_traj_new from tools. "
            "Run from the repo root or check your workspace layout."
        ) from exc
    preprocess_main()


if __name__ == "__main__":
    main()
