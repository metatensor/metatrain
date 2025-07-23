<!-- Describe your new architecture briefly here -->



# Contributor (creator of pull-request) checklist

- [ ] Add your architecture to the `experimental` or `stable` folder. See the
  [docs/src/dev-docs/architecture-life-cycle.rst](Architecture life cycle)
  document for requirements. `src/metatrain/experimental/<architecture_name>`
- [ ] Add default hyperparameter file to
  `src/metatrain/experimental/<architecture_name>/default-hypers.yml`
- [ ] Add your architecture to the CI in `.github/workflow/architecture-tests.yml`
- [ ] Add a new dependencies entry in the `optional-dependencies` section in the
  `pyproject.toml`
- [ ] Add tests:
  - [ ] checking that the code is compatible with TorchScript
  - [ ] checking the basic functionality (invariance, fitting, prediction)
  - [ ] checking that the checkpoints are properly versionned (see the existing
    `test_checkpoint.py` in other architectures)
- [ ] Add maintainers as codeowners in [CODEOWNERS](CODEOWNERS)
- [ ] Trigger a final GPU test by commenting "cscs-ci run" in the PR. Only do this after
  all other checks are approved and the PR is ready to merge.

# Reviewer checklist

## New experimental architectures

- [ ] Capability to fit at least a single quantity and predict it, verified through CI
   tests.
- [ ] Compatibility with JIT compilation using `TorchScript
   <https://pytorch.org/docs/stable/jit.html>`_.
- [ ] Provision of reasonable default hyperparameters.
- [ ] A contact person designated as the maintainer, mentioned in `__maintainers__` and the `CODEOWNERS` file
- [ ] All external dependencies must be pip-installable. While not required to be on
   PyPI, a public git repository or another public URL with a repository is acceptable.


## New stable architectures
- [ ] Provision of regression prediction tests with a small (not exported) checkpoint
  file.
- [ ] Comprehensive architecture documentation
- [ ] If an architecture has external dependencies, all must be publicly available on
  PyPI.
- [ ] Adherence to the standard output infrastructure of `metatrain`, including
   logging and model save locations.
