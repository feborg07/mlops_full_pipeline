
from pathlib import Path

# ============================================================
# Sprint 17 - MLOps Project Structure
# Cross-platform: Windows, macOS, Linux
# ============================================================

# Project root (relative to this script's location) make sure you run this file on the parent folder of your project
ROOT = Path(__file__).resolve().parent / "mlops-final-project"

# Directories
directories = [
    ".github/workflows",
    "config",
    "data/raw",
    "data/processed",
    "data/drift",
    "notebooks",
    "src",
    "tests",
    "models",
    "reports/figures",
    "reports/drift",
    "scripts",
]

# Files
files = [
    ".github/workflows/ci.yml",
    ".github/workflows/retrain.yml",
    "config/config.yaml",
    "params.yaml",
    "requirements.txt",
    "pytest.ini",
    ".gitignore",
    ".dvcignore",
    "README.md",
    "notebooks/01_eda.ipynb",
    "src/__init__.py",
    "src/preprocess.py",
    "src/train.py",
    "src/evaluate.py",
    "src/predict.py",
    "src/monitoring.py",
    "tests/test_preprocess.py",
    "tests/test_train.py",
    "tests/test_monitoring.py",
    "scripts/generate_drift.py",
]

# Create directories
for directory in directories:
    (ROOT / directory).mkdir(parents=True, exist_ok=True)

# Create files without overwriting existing content
for file in files:
    (ROOT / file).touch(exist_ok=True)

# Preserve empty directories in Git
for directory in [
    "data/raw",
    "data/processed",
    "data/drift",
    "models",
    "reports/figures",
    "reports/drift",
]:
    (ROOT / directory / ".gitkeep").touch(exist_ok=True)

print(f"Project created successfully at: {ROOT}")
