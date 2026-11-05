
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

# ============================================================
# Sprint 17 - Generate requirements.txt
# ============================================================

# Project root directory - change it if necessary
ROOT = Path(__file__).resolve().parent

# Direct dependencies used by our MLOps project
packages = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "mlflow",
    "dvc",
    "evidently",
    "pytest",
    "PyYAML",
    "jupyterlab",
    "ipykernel",
]

# Collect installed package versions
requirements = []
missing = []

for package in packages:
    try:
        installed_version = version(package)
        requirements.append(f"{package}=={installed_version}")
    except PackageNotFoundError:
        missing.append(package)

# Do not silently create an incomplete requirements file
if missing:
    print("Missing packages:")
    for package in missing:
        print(f"  - {package}")
    raise SystemExit(
        "Install missing packages before generating requirements.txt"
    )

# Write requirements.txt
output = ROOT / "requirements.txt"
output.write_text(
    "\n".join(requirements) + "\n",
    encoding="utf-8"
)

print(f"Generated: {output}")
print("\n".join(requirements))
