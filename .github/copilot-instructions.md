# Copilot Instructions

Metatrain is a **CLI for training & evaluating machine learning models for atomistic systems**. It provides a unified YAML-based interface (`mtt train options.yaml`) for multiple ML architectures (SOAP-BPNN, PET, GAP, MACE, FlashMD, LLPR) with automatic TorchScript export compatibility for MD engines (LAMMPS, ASE, i-PI). New architectures follow a standardized plugin system pattern where each architecture is discoverable via `metatrain.utils.architectures.import_architecture()`.

## Repository Structure

| Path                                     | Description                                              |
| -----------------------------------------| -------------------------------------------------------- |
| `src/metatrain/`                         | Python source code                                       |
| `src/metatrain/cli/`                     | Command-line interface (train, eval, export)             |
| `src/metatrain/utils/`                   | Shared utilities (data, I/O, losses, testing)            |
| `src/metatrain/{soap_bpnn,pet,gap,...}/` | Architecture implementations (plugin system)             |
| `src/metatrain/experimental/`            | Experimental architectures (e.g., mace, flashmd)         |
| `src/metatrain/deprecated/`              | Deprecated architectures (e.g., nanopet)                 |
| `tests/`                                 | Test suite using pytest                                  |
| `examples/`                              | Training examples and YAML configuration templates       |
| `docs/`                                  | Sphinx documentation in reStructuredText                 |

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

**Development Tip**: Test setup (generating reference outputs via `bash {toxinidir}/tests/resources/generate-outputs.sh`) can take a while. For faster iteration during development, you can temporarily comment out this step in `tox.ini` under `[testenv:tests]` → `commands_pre`, then remember to uncomment before final testing.

## Training & Export Pipeline

The core workflow follows this sequence:

1. Parse YAML config with `OmegaConf` and validate via `utils/pydantic.py`
2. Dynamic architecture loading: `import_architecture(name)` returns module with `__model__` and `__trainer__`
3. Load datasets: `metatrain.utils.data.get_dataset()` reads XYZ files via `ase.io.read()`
4. Training: `Trainer.train()` runs forward passes, loss computed via `utils/loss.py`
5. Checkpoint: Model saved with version tracking (`__checkpoint_version__`)
6. Export: `export_model()` converts to TorchScript for MD engines (LAMMPS, ASE, i-PI)

## General Coding Guidelines

- **Imports**: Use absolute imports only (`from metatrain.utils...`), never relative imports
- **Type hints**: Required in `src/metatrain/cli/`, `src/metatrain/utils/`, and architecture folders (enforced by `lint_strict_folders` in `tox.ini`)
- **Docstrings**: All public functions must have docstrings. Use `pydoclint` checks on strict folders
- **Errors**: Raise custom exceptions from `utils/errors.py` (e.g., `ArchitectureError`, `OutOfMemoryError`)
- **Architecture pattern**: Each architecture (`src/metatrain/{name}/`) must include:
  - `__init__.py`: Exports `__model__` and `__trainer__` class references
  - `documentation.py`: Defines `ModelHypers` and `TrainerHypers` Pydantic classes
  - `model.py`: Implements `ModelInterface` from `utils/abc.py`
  - `trainer.py`: Implements `TrainerInterface` from `utils/abc.py`
- **YAML validation**: Use `OmegaConf` → Pydantic validators in `utils/pydantic.py` → `check_architecture_options()`
- **Data handling**: 
  - Input: Extended XYZ files via `ase.io.read()`
  - Targets: `metatensor.TensorMap` (sparse tensor format for properties)
  - Metadata: `DatasetInfo` tracks atomic types and per-target statistics
- **Checkpoint versioning**: Increment `__checkpoint_version__` and implement `Model.upgrade_checkpoint()` for backward compatibility
- **Device/dtype support**: Define `__supported_devices__` and `__supported_dtypes__` class attributes on Model
- **Code style**: Handled by ruff (format + check), yamlfix, prettier. Run `tox -e format` before committing

## Versioning

Metatrain uses semantic versioning (e.g., `2025.3`). Model checkpoints are versioned separately via `__checkpoint_version__` class attribute to handle breaking changes while maintaining backward compatibility through upgrade logic.

## Pull Request Guidelines

Before submitting a pull request, ensure:

- Code is formatted: Run `tox -e format` and commit changes
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

- Imports are absolute, not relative
- Type hints are present in strict folders (`lint_strict_folders`: `cli/`, `utils/`, architecture folders)
- Docstrings follow existing patterns and match function signatures
- Error handling uses custom exceptions from `utils/errors.py`
- YAML schema changes are validated through Pydantic
- Architecture changes implement required abstract methods from `ModelInterface` or `TrainerInterface`
- Checkpoint version bumps include upgrade logic in `upgrade_checkpoint()`
- Tests cover both success and edge cases
- Documentation is updated for user-facing changes (e.g., new YAML options, CLI flags)
- All `tox` checks pass (verified in CI, but reviewers can validate locally)
