"""Standalone Script: Clean or Quarantine Empty YOLO Annotation Files.

Detects 0-byte and whitespace-only .txt files.
Offers safe quarantine into a separate folder or confirmed deletion.
"""

import argparse
import shutil
from pathlib import Path


def clean_empty(folder: Path, quarantine_dir: Path | None = None, delete: bool = False) -> None:
    txt_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt" and p.name.lower() != "classes.txt"]
    empty_files = []

    for txt in txt_files:
        if txt.stat().st_size == 0 or not txt.read_text(encoding="utf-8", errors="ignore").strip():
            empty_files.append(txt)

    if not empty_files:
        print(f"No empty or whitespace-only .txt files found in {folder}.")
        return

    print(f"Found {len(empty_files)} empty or invalid .txt file(s).")

    if quarantine_dir is not None:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for f in empty_files:
            shutil.move(str(f), str(quarantine_dir / f.name))
        print(f"Quarantined {len(empty_files)} empty file(s) to: {quarantine_dir}")

    elif delete:
        confirm = input(f"Are you sure you want to PERMANENTLY delete {len(empty_files)} files from {folder}? [y/N]: ")
        if confirm.strip().lower() in ("y", "yes"):
            for f in empty_files:
                f.unlink()
            print(f"Deleted {len(empty_files)} empty file(s).")
        else:
            print("Deletion cancelled.")
    else:
        print("Empty files found (dry-run):")
        for f in empty_files[:15]:
            print(f"  {f.name}")
        if len(empty_files) > 15:
            print(f"  ... and {len(empty_files) - 15} more.")
        print("\nTip: Run with --quarantine path\\to\\quarantine or --delete to take action.")


def main():
    parser = argparse.ArgumentParser(description="Find, quarantine, or delete empty/whitespace YOLO labels.")
    parser.add_argument("folder", type=Path, help="Folder containing .txt files")
    parser.add_argument("--quarantine", type=Path, default=None, help="Move empty files to this folder")
    parser.add_argument("--delete", action="store_true", help="Permanently delete empty files (asks confirmation)")
    args = parser.parse_args()

    clean_empty(args.folder, args.quarantine, args.delete)


if __name__ == "__main__":
    main()
