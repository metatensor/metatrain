<!-- Describe your new architecture briefly here -->



# Contributor (creator of pull-request) checklist

- [ ] Add your architecture to the experimental/stable folder. See the
  [docs/src/dev-docs/architecture-life-cycle.rst](Architecture life cycle) document for
  requirements.
  `src/metatrain/experimental/<architecture_name>`
- [ ] Add default hyperparameter file to
  `src/metatrain/experimental/<architecture_name>/default-hypers.yml`
- [ ] Add a `.yml` file into github workflows `.github/workflow/<architecture_name>.yml`
- [ ] Architecture dependencies entry in the `optional-dependencies` section in the
  `pyproject.toml`
- [ ] Tests: torch-scriptability, basic functionality (invariance, fitting, prediction)
- [ ] Add maintainers as codeowners in [CODEOWNERS](CODEOWNERS)

# Reviewer checklist

## New experimental architectures

- [ ] Capability to fit at least a single quantity and predict it, verified through CI
   tests.
- [ ] Compatibility with JIT compilation using `TorchScript
   <https://pytorch.org/docs/stable/jit.html>`_.
- [ ] Provision of reasonable default hyperparameters.
- [ ] A contact person designated as the maintainer.
- [ ] All external dependencies must be pip-installable. While not required to be on
   PyPI, a public git repository or another public URL with a repository is acceptable.


## New stable architectures
- [ ] Provision of regression prediction tests with a small (not exported) checkpoint
  file.
- [ ] Comprehensive architecture documentation
- [ ] If an architecture has external dependencies, all must be publicly available on
  PyPI.
- [ ] Adherence to the standard output infrastructure of `metatensor-models`, including
   logging and model save locations.
