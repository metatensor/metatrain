#!/usr/bin/env python3
"""
JsonFix: Lint and format JSON files.

Recursively traverses given files and directories, validates JSON syntax, and reformats
files with consistent indentation.
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :returns: Parsed arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to process (.json files only)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if files are valid and properly formatted (no changes made)",
    )
    return parser.parse_args()


def collect_json_files(paths: list[str]) -> list[Path]:
    """
    Recursively collect all .json files from the given paths.

    :param paths: List of file and directory paths
    :returns: List of resolved JSON file paths
    """
    json_files: list[Path] = []

    for entry in paths:
        path = Path(entry)

        if path.is_file() and path.suffix == ".json":
            json_files.append(path.resolve())
        elif path.is_dir():
            json_files.extend(f.resolve() for f in path.rglob("*.json") if f.is_file())

    return json_files


def check_and_format_file(path: Path, check_only: bool = False) -> tuple[bool, bool]:
    """
    Check and (optionally) format a single JSON file.

    :param path: Path to the JSON file.
    :param check_only: If True, only check; otherwise, reformat if needed.
    :returns: is_valid and was_changed
    """
    try:
        original = path.read_text(encoding="utf-8")
        data = json.loads(original)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False, False

    formatted = json.dumps(data, indent=4, ensure_ascii=False) + "\n"

    if original != formatted:
        if not check_only:
            path.write_text(formatted, encoding="utf-8")
        return True, True

    return True, False


def main() -> None:
    """Main function coordinating JSON file checking and formatting."""
    args = parse_args()
    json_files = collect_json_files(args.paths)

    print("JsonFix: Checking files" if args.check else "JsonFix: Fixing files")

    total = 0
    changed = 0
    had_invalid = False

    for file_path in json_files:
        is_valid, was_changed = check_and_format_file(file_path, check_only=args.check)
        total += 1

        if not is_valid:
            print(f"✘ Invalid JSON: {file_path}")
            had_invalid = True
            continue

        if was_changed:
            action = "Would fix" if args.check else "Fixed"
            print(f"{action} {file_path}")
            changed += 1

    print(f"Checked {total} files: {changed} fixed, {total - changed} left unchanged")

    if had_invalid or (args.check and changed > 0):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
