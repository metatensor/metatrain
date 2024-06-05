<!-- Describe your new architecture briefly here -->



# New architecture TODOs

- [ ] Add your architecture to the experimental folder
  `src/metatrain/experimental/<architecture_name>`
- [ ] Add default hyperparameter file to
  `src/metatrain/cli/conf/architecture/experimental.<architecture_name>.yaml`
- [ ] Add a `.yml` file into github workflows `.github/workflow/<architecture_name>.yml`
- [ ] Architecture dependencies entry in the `optional-dependencies` section in the
  `pyproject.toml`
- [ ] Tests: torch-scriptability, basic functionality (invariance, fitting, prediction)
