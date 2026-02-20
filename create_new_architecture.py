#!/usr/bin/env python3
"""
Create a new independent architecture in metatrain.

This script copies the vanilla PET architecture and creates a new architecture
with the given name, automatically updating all imports.

Usage:
    python create_new_architecture.py pet_1
"""

import argparse
import shutil
import sys
from pathlib import Path
import re


def update_imports_in_file(filepath: Path, old_arch: str, new_arch: str) -> None:
    """Update imports from old_arch to new_arch in a Python file."""
    try:
        content = filepath.read_text()
        
        # Replace imports
        content = re.sub(
            rf'from metatrain\.{re.escape(old_arch)}\.', 
            f'from metatrain.{new_arch}.', 
            content
        )
        content = re.sub(
            rf'from metatrain\.{re.escape(old_arch)} import',
            f'from metatrain.{new_arch} import',
            content
        )
        content = re.sub(
            rf'import metatrain\.{re.escape(old_arch)}\.',
            f'import metatrain.{new_arch}.',
            content
        )

        # Update architecture_name entries inside checkpoints
        content = re.sub(
            r'"architecture_name"\s*:\s*"pet"',
            f'"architecture_name": "{new_arch}"',
            content,
        )
        content = re.sub(
            r"'architecture_name'\s*:\s*'pet'",
            f"'architecture_name': '{new_arch}'",
            content,
        )
        
        filepath.write_text(content)
    except Exception as e:
        print(f"Warning: Could not update {filepath}: {e}")


def create_architecture(arch_name: str) -> None:
    """Create a new architecture by copying and modifying vanilla PET."""
    
    # Normalize path
    metatrain_src = Path("./src").resolve()
    
    # Determine paths
    source_arch = metatrain_src / "metatrain" / "pet"
    
    # Handle nested names like "experimental.my_arch"
    arch_parts = arch_name.split(".")
    target_arch = metatrain_src / "metatrain" / Path(*arch_parts)
    
    # ========================================================================
    # Validation
    # ========================================================================
    
    if not source_arch.exists():
        print(f"❌ Error: Cannot find source PET architecture at: {source_arch}")
        sys.exit(1)
    
    if target_arch.exists():
        print(f"❌ Error: Architecture '{arch_name}' already exists at: {target_arch}")
        sys.exit(1)
    
    # ========================================================================
    # Create directory structure
    # ========================================================================
    
    print(f"Creating architecture: {arch_name}")
    print(f"Source: {source_arch}")
    print(f"Target: {target_arch}")
    print()
    
    (target_arch / "modules").mkdir(parents=True, exist_ok=True)
    (target_arch / "tests").mkdir(parents=True, exist_ok=True)
    
    print("✓ Created directories")
    
    # ========================================================================
    # Copy files
    # ========================================================================
    
    # Copy main files
    for filename in ["__init__.py", "model.py", "trainer.py", "documentation.py"]:
        src = source_arch / filename
        dst = target_arch / filename
        if src.exists():
            shutil.copy2(src, dst)
    
    print("✓ Copied main files (__init__.py, model.py, trainer.py, documentation.py)")
    
    # Copy modules
    modules_src = source_arch / "modules"
    if modules_src.exists():
        for item in modules_src.iterdir():
            if item.is_file():
                shutil.copy2(item, target_arch / "modules" / item.name)
            elif item.is_dir():
                shutil.copytree(item, target_arch / "modules" / item.name, dirs_exist_ok=True)
    
    print("✓ Copied modules directory")
    
    # Copy optional files
    if (source_arch / "checkpoints.py").exists():
        shutil.copy2(source_arch / "checkpoints.py", target_arch / "checkpoints.py")
        print("✓ Copied checkpoints.py")
    
    # if (source_arch / "tests").exists():
    #     for item in (source_arch / "tests").iterdir():
    #         if item.is_file():
    #             shutil.copy2(item, target_arch / "tests" / item.name)
    #         elif item.is_dir():
    #             shutil.copytree(item, target_arch / "tests" / item.name, dirs_exist_ok=True)
    #     print("✓ Copied tests directory")
    
    # ========================================================================
    # Update imports
    # ========================================================================
    
    print()
    print(f"Updating imports from 'metatrain.pet' to 'metatrain.{arch_name}'...")
    
    for python_file in target_arch.rglob("*.py"):
        update_imports_in_file(python_file, "pet", arch_name)
    
    print("✓ Updated imports in all Python files")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print()
    print(f"✅ Successfully created architecture: {arch_name}")
    print()
    print(f"Location: {target_arch}")
    print()
    print("Next steps:")
    print(f"1. Modify the model in: {target_arch / 'model.py'}")
    print(f"2. Customize hyperparameters in: {target_arch / 'documentation.py'}")
    print("3. Test the architecture:")
    print(f"   python -c \"from metatrain.utils.architectures import import_architecture; arch = import_architecture('{arch_name}'); print('Architecture loaded successfully!')\"")
    print()
    print("4. Use in training with options.yaml:")
    print("   architecture:")
    print(f"     name: {arch_name}")
    print("     model:")
    print("       cutoff: 5.0")
    print("       # ... other hyperparameters")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Create a new independent architecture in metatrain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_new_architecture.py pet_1
  python create_new_architecture.py pet_2
  python create_new_architecture.py experimental.my_arch
  python create_new_architecture.py pet_custom /path/to/metatrain/src
        """,
    )
    
    parser.add_argument(
        "name",
        help="Name of the new architecture (e.g., pet_1, experimental.my_arch)",
    )
    
    args = parser.parse_args()
    
    try:
        create_architecture(args.name)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
