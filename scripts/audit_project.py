
"""
Sprint 17 - Project File Audit

Recursively inspect the repository for:
    - Empty files (0 bytes)
    - Whitespace-only text files
    - Empty directories
    - Potentially obsolete files

Read-only: no files or directories are modified.

Usage:
    python scripts/audit_project.py
"""

from pathlib import Path

# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".vscode",
    "node_modules",
    "site-packages",
    "mlruns",
}

EXCLUDED_PATHS = {
    Path(".dvc/cache"),
    Path(".dvc/tmp"),
}

TEXT_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".sh",
}

BACKUP_SUFFIXES = {
    ".bak",
    ".backup",
    ".old",
    ".orig",
    ".tmp",
    ".temp",
    ".swp",
    ".swo",
}


# ============================================================
# Helpers
# ============================================================

def is_excluded(path: Path) -> bool:
    """Determine whether a path should be skipped."""

    relative = path.relative_to(ROOT)

    if any(
        part in EXCLUDED_DIRS
        for part in relative.parts
    ):
        return True

    return any(
        relative == excluded
        or excluded in relative.parents
        for excluded in EXCLUDED_PATHS
    )


def is_whitespace_only(path: Path) -> bool:
    """Check whether a text file contains only whitespace."""

    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False

    try:
        return not path.read_text(
            encoding="utf-8"
        ).strip()

    except (UnicodeError, OSError):
        return False


def is_potential_backup(path: Path) -> bool:
    """Identify common backup and temporary filenames."""

    name = path.name.lower()

    return (
        path.suffix.lower() in BACKUP_SUFFIXES
        or name.endswith("~")
        or name.startswith("copy of ")
    )


# ============================================================
# Audit
# ============================================================

def audit_project(root: Path = ROOT) -> dict:
    """Recursively inspect project files and directories."""

    import os

    root = Path(root).resolve()

    empty_files = []
    whitespace_files = []
    empty_directories = []
    potential_backups = []

    total_files = 0
    total_directories = 0

    for current, directories, filenames in os.walk(root):

        current_path = Path(current)

        # Prevent traversal into excluded directories.
        directories[:] = sorted(
            directory
            for directory in directories
            if not is_excluded(
                current_path / directory
            )
        )

        filenames.sort()
        total_directories += 1

        # Empty means no included files or subdirectories.
        included_files = [
            name
            for name in filenames
            if not is_excluded(current_path / name)
        ]

        if (
            current_path != root
            and not directories
            and not included_files
        ):
            empty_directories.append(current_path)

        for filename in included_files:

            path = current_path / filename

            if path.is_symlink() or not path.is_file():
                continue

            total_files += 1

            try:
                size = path.stat().st_size
            except OSError:
                continue

            if size == 0:
                empty_files.append(path)

            elif is_whitespace_only(path):
                whitespace_files.append(path)

            if is_potential_backup(path):
                potential_backups.append(path)

    return {
        "root": root,
        "total_files": total_files,
        "total_directories": total_directories,
        "empty_files": empty_files,
        "whitespace_files": whitespace_files,
        "empty_directories": empty_directories,
        "potential_backups": potential_backups,
    }


# ============================================================
# Reporting
# ============================================================

def print_section(title: str, paths: list, root: Path):
    """Print repository-relative paths."""

    print(f"\n{title} ({len(paths)})")
    print("-" * 65)

    if not paths:
        print("None")
        return

    for path in paths:
        print(path.relative_to(root).as_posix())


def main():

    results = audit_project()

    root = results["root"]

    print("\n" + "=" * 65)
    print("SPRINT 17 - PROJECT FILE AUDIT")
    print("=" * 65)

    print(f"Project: {root}")
    print(f"Files inspected: {results['total_files']}")
    print(f"Directories inspected: {results['total_directories']}")

    print_section(
        "EMPTY FILES (0 BYTES)",
        results["empty_files"],
        root,
    )

    print_section(
        "WHITESPACE-ONLY FILES",
        results["whitespace_files"],
        root,
    )

    print_section(
        "EMPTY DIRECTORIES",
        results["empty_directories"],
        root,
    )

    print_section(
        "POTENTIAL BACKUP / TEMPORARY FILES",
        results["potential_backups"],
        root,
    )

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE - NO FILES MODIFIED")
    print("=" * 65)


if __name__ == "__main__":
    main()
