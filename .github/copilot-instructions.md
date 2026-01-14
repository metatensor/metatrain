# Copilot Instructions

Metatrain is a **CLI for training & evaluating machine learning models for atomistic systems**. It provides a unified YAML-based interface (`mtt train options.yaml`) for multiple ML architectures (SOAP-BPNN, PET, GAP, MACE, FlashMD, LLPR) with automatic TorchScript export compatibility for MD engines (LAMMPS, ASE, i-PI). New architectures follow a standardized plugin system pattern where each architecture is discoverable via `metatrain.utils.architectures.import_architecture()`.

## Repository Structure

| Path                                     | Description                                              |
| ---------------------------------------- | -------------------------------------------------------- |
| `.github/`                               | GitHub workflows and issue templates                     |
| `developer/`                             | Helper scripts and tools for development                 |
| `docs/`                                  | Sphinx documentation in reStructuredText                 |
| `examples/`                              | Training examples and YAML config templates rendered     |
|                                          | with sphinx-gallery for the documentation                |
| `src/metatrain/`                         | Python source code                                       |
| `src/metatrain/cli/`                     | Command-line interface (train, eval, export)             |
| `src/metatrain/deprecated/`              | Deprecated architectures (e.g., nanopet)                 |
| `src/metatrain/experimental/`            | Experimental architectures (e.g., mace, flashmd)         |
| `src/metatrain/{soap_bpnn,pet,gap,...}/` | Stable architecture implementations (plugin system)      |
| `src/metatrain/utils/`                   | Shared utilities (data, I/O, losses, testing)            |
| `tests/`                                 | Test suite using pytest                                  |
| `CITATION.cff`                           | Citation metadata for the project                        |
| `CODEOWNERS`                             | GitHub code owners for review assignment                 |
| `CONTRIBUTING.rst`                       | Contributing guidelines in reStructuredText              |
| `LICENSE`                                | BSD-3-Clause license                                     |
| `MANIFEST.in`                            | Files to include in package distribution                 |
| `README.md`                              | Project overview and quick start                         |
| `pyproject.toml`                         | Python project configuration and dependencies            |
| `tox.ini`                                | Testing and development environment configuration        |

## Environment Setup

Install dependencies and setup for development:

```bash
pip install tox
git clone https://github.com/metatensor/metatrain
cd metatrain
pip install -e .
```

To see all available tox environments:

```bash
tox list
```

## Testing

All tests are executed via `tox`. Common commands:

```bash
tox                  # Run all tests (lint, format, tests, architecture tests)
tox -e tests         # Core library unit tests only
tox -e lint          # Code style, type hints, docstrings
tox -e format        # Auto-format code (ruff, yamlfix, prettier)
tox -e {arch}-tests  # Architecture-specific tests (e.g., tox -e soap-bpnn-tests)
tox -e docs          # Build documentation
tox -e tests -- tests/utils/test_abc.py  # Run specific test file
tox -e tests -- tests/test_init.py::test_some_function  # Run specific test function
```

Tests use `pytest` framework with fixtures in `tests/resources/`. Architecture tests inherit from `metatrain.utils.testing.ArchitectureTests` which provides reusable test classes (Input, Output, Training, Checkpoints, TorchScript, Exported).

**Development Tip**:

- Test setup (generating reference outputs via `bash {toxinidir}/tests/resources/generate-outputs.sh`) can take a while. For faster iteration during development, you can temporarily comment out this step in `tox.ini` under `[testenv:tests]` → `commands_pre`, then remember to uncomment before final testing.
- tox creates a private .tox/ folder for virtual environments. To avoid issues with cached dependencies, you can delete this folder and re-run `tox` to recreate clean environments.

## Training & Export Pipeline

The core workflow follows this sequence:

1. Parse YAML config with `OmegaConf` and validate via `utils/pydantic.py`
2. Dynamic architecture loading: `import_architecture(name)` returns module with `__model__` and `__trainer__`
3. Load datasets: `metatrain.utils.data.get_dataset()` reads XYZ files via `ase.io.read()`
4. Training: `Trainer.train()` runs forward passes, loss computed via `utils/loss.py`
5. Checkpoint: Model saved with version tracking (`__checkpoint_version__`)
6. Export: `export_model()` converts to TorchScript for MD engines (LAMMPS, ASE, i-PI)

## General Coding Guidelines

- **Line length**: Use 88 character for comments and multi line strings in **all** files. Code lines will be formatted by the formatter.
- **Imports**: Use relative imports in library (`from .utils...`), and absolute imports in tests
- **Type hints**: Required in `src/metatrain/cli/`, `src/metatrain/utils/`, and architecture folders (enforced by `lint_strict_folders` in `tox.ini`)
- **Docstrings**: All public functions must have docstrings. Use `pydoclint` checks on strict folders
- **Errors**: Raise custom exceptions from `utils/errors.py` (e.g., `ArchitectureError`, `OutOfMemoryError`)
- **Architecture pattern**: Each architecture (`src/metatrain/{name}/`) must include:
  - `__init__.py`: Exports `__model__` and `__trainer__` class references
  - `documentation.py`: Defines `ModelHypers` and `TrainerHypers` Pydantic classes
  - `model.py`: Implements `ModelInterface` from `utils/abc.py`
  - `trainer.py`: Implements `TrainerInterface` from `utils/abc.py`
- **Data handling**: 
  - Input: Extended XYZ files via `ase.io.read()`
  - Targets: `metatensor.TensorMap` (sparse tensor format for properties)
  - Metadata: `DatasetInfo` tracks atomic types and per-target statistics
- **Checkpoint versioning**: Increment `__checkpoint_version__` and implement `Model.upgrade_checkpoint()` for backward compatibility
- **Device/dtype support**: Define `__supported_devices__` and `__supported_dtypes__` class attributes on Model
- **Code style**: Handled by ruff (format + check), yamlfix, prettier. Run `tox -e format` before committing

## Versioning

Metatrain versioning based on year and month of the release (e.g., `2025.3`). Since metatrain is not a library we don't follow semantic versioning for public APIs and even a "minor" release can introduce breaking changes.
Model checkpoints are versioned separately via `__checkpoint_version__` class attribute to handle breaking changes while maintaining backward compatibility through upgrade logic.

## Pull Request Guidelines

Before submitting a pull request, ensure:

- Code is formatted: Run `tox -e format` and `tox -e lint` passes before committing changes
- All `tox` checks pass locally: `tox` (or specific environments like `tox -e tests`)
- Type hints and docstrings are present in strict folders (`cli/`, `utils/`, architecture folders)
- Tests pass: `tox -e tests` for core tests, `tox -e {arch}-tests` for architecture tests
- Documentation is updated if functionality changes: `docs/` uses reStructuredText
- PR description includes:
  - Summary of changes
  - Any new dependencies added
  - Links to relevant issues or discussions
  - Context for reviewers (e.g., "fixes #1234")

## Code Review Guidelines

When reviewing code, verify:

- Type hints are present in strict folders (`lint_strict_folders`: `cli/`, `utils/`, architecture folders)
- Docstrings follow existing patterns and match function signatures
- Error handling uses custom exceptions from `utils/errors.py`
- YAML schema changes are validated through Pydantic
- Architecture changes implement required abstract methods from `ModelInterface` or `TrainerInterface`
- Checkpoint version bumps include upgrade logic in `upgrade_checkpoint()`
- Tests cover both success and edge cases
- Documentation is updated for user-facing changes (e.g., new YAML options, CLI flags)
- All `tox` checks pass (verified in CI, but reviewers can validate locally)
