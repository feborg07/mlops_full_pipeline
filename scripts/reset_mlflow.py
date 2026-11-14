
"""Reset local MLflow experiments and generated training artifacts.

Usage:
    python -m scripts.reset_mlflow
    python -m scripts.reset_mlflow --yes
"""

import argparse  # Parse command-line arguments.
import shutil  # Remove directories recursively.
from pathlib import Path  # Manage filesystem paths.


# scripts/reset_mlflow.py is one directory below the project root.
ROOT = Path(__file__).resolve().parents[1]

# Only delete generated MLflow and training files.
RESET_PATHS = [
    ROOT / "mlflow.db",
    ROOT / "mlflow.db-shm",
    ROOT / "mlflow.db-wal",
    ROOT / "mlruns",
    ROOT / "mlartifacts",
    ROOT / "models" / "training",
]


def reset_mlflow(skip_confirmation=False):
    """Delete local experiment history and generated model artifacts."""

    print("\nPaths selected for deletion:\n")

    for path in RESET_PATHS:
        status = "EXISTS" if path.exists() else "NOT FOUND"
        print(f"[{status}] {path.relative_to(ROOT)}")

    # Require confirmation unless --yes was provided.
    if not skip_confirmation:
        answer = input("\nDelete these paths permanently? [y/N]: ")

        if answer.strip().lower() != "y":
            print("Reset cancelled.")
            return

    for path in RESET_PATHS:
        # Remove symbolic links without following their targets.
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    print("\nMLflow reset completed.")


def main():
    """Parse CLI arguments and execute cleanup."""

    parser = argparse.ArgumentParser(
        description="Reset local MLflow experiments."
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation.",
    )

    args = parser.parse_args()

    reset_mlflow(skip_confirmation=args.yes)


if __name__ == "__main__":
    main()
