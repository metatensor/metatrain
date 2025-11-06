<h1 align="center">
    <img src="https://raw.githubusercontent.com/metatensor/metatrain/refs/heads/main/docs/src/logo/metatrain-horizontal-dark.svg" alt="Metatensor logo" width="600"/>
</h1>

<h4 align="center">

[![tests status](https://img.shields.io/github/checks-status/metatensor/metatrain/main)](https://github.com/metatensor/metatrain/actions?query=branch%3Amain)
[![documentation](https://img.shields.io/badge/📚_documentation-latest-sucess)](https://metatensor.github.io/metatrain)
[![coverage](https://codecov.io/gh/metatensor/metatrain/branch/main/graph/badge.svg)](https://codecov.io/gh/metatensor/metatrain)
</h4>

<!-- marker-introduction -->

## What is metatrain?

`metatrain` is a user-friendly command line tool that helps you **train machine learning
models for atomistic systems**. In simple terms, it lets you create models that can
predict properties of molecules and materials (like energy and forces) by learning from
examples.

### Why use machine learning for atomistic systems?

Traditional quantum mechanical calculations (like DFT) can be very accurate but extremely
slow, especially for large systems or long simulations. Machine learning models can learn
from these accurate calculations and then make predictions thousands of times faster,
while maintaining good accuracy. This enables simulations that would otherwise be
impossible.

### What does metatrain do?

- **Training**: Teaches machine learning models using your data (atomic structures and
  their properties)
- **Evaluation**: Tests how well your trained model performs
- **Export**: Saves models that can be used directly in molecular dynamics simulations
  with engines like LAMMPS, i-PI, or ASE

Everything is configured through simple text files (YAML format), so you don't need to
write code to use it. Once trained, your models work seamlessly with various simulation
engines through the [metatomic](https://docs.metatensor.org/metatomic) interface.

### Do I need to know machine learning?

**No!** This documentation will guide you through the process step by step. We'll explain
concepts as we go, assuming you have basic familiarity with ASE (Atomic Simulation
Environment) for working with atomic structures.

> **Note**: `metatrain` focuses on making training and evaluation easy. The actual
> machine learning algorithms come from separate architecture packages that plug into
> metatrain.

<!-- marker-architectures -->

# List of Implemented Architectures

Currently `metatrain` supports the following architectures for building an atomistic
model:

| Name                     | Description                                                                                                                          |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| GAP                      | Sparse Gaussian Approximation Potential (GAP) using Smooth Overlap of Atomic Positions (SOAP).                                       |
| PET                      | Point Edge Transformer (PET), interatomic machine learning potential                                                                 |
| NanoPET *(deprecated)*   | Re-implementation of the original PET with slightly improved training and evaluation speed                                           |
| SOAP BPNN                | A Behler-Parrinello neural network with SOAP features                                                                                |
| FlashMD                  | An architecture for the direct prediction of molecular dynamics                                                                      |

<!-- marker-documentation -->

# Documentation

For details, tutorials, and examples, please visit our
[documentation](https://metatensor.github.io/metatrain/latest/).

<!-- marker-installation -->

# Installation

## Quick install (recommended for beginners)

The easiest way to get started is with pip:

```bash
pip install metatrain
```

This installs the core `metatrain` package. You'll also need to install at least one
machine learning architecture. For beginners, we recommend starting with SOAP-BPNN:

```bash
pip install metatrain[soap-bpnn]
```

**Why SOAP-BPNN?** It's fast to train, works well for many systems, and doesn't require
GPU acceleration (though it can use one if available).

## Alternative: conda installation

If you prefer conda:

```bash
conda install -c conda-forge metatrain
```

> ⚠️ **Note**: The conda installation includes only basic architectures like PET. For
> other architectures, you may need additional pip installations.

## Verify installation

After installation, verify everything works:

```bash
mtt --help
```

You should see a list of available commands. You're now ready to train models!

<!-- marker-quickstart -->

# Quickstart

Once installed, you can train a machine learning model with a single command:

```bash
mtt train options.yaml
```

The `options.yaml` file tells metatrain what to do. Here's a simple example that trains
a model to predict energies:

```yaml
# Choose which machine learning architecture to use
architecture:
  name: soap_bpnn  # A neural network with SOAP descriptors - good for beginners
  
training:
  num_epochs: 5  # How many times to go through the training data
                 # (5 is very short - just for testing; typically use 50-200)

# Where is your training data?
training_set:
  systems: "qm9_reduced_100.xyz"  # File with atomic structures (positions, elements, cell)
  targets:
    energy:                        # We want to predict energy
      key: "U0"                    # Name of energy column in your data file
      unit: "eV"                   # Energy units in your data

# Automatically split data for validation and testing
validation_set: 0.1  # Hold out 10% to monitor training progress
test_set: 0.1        # Hold out 10% to evaluate final performance
```

**What's happening here?**
- **Training set**: Examples the model learns from (80% of data)
- **Validation set**: Used during training to check if model is improving (10% of data)
- **Test set**: Used after training to see how well the model performs on unseen data (10% of data)
- **Epochs**: One epoch = the model sees all training examples once. More epochs = more learning (but too many can cause overfitting)

After running this, you'll have a trained model file (`model.pt`) ready to use!

<!-- marker-shell -->

# Shell Completion

`metatrain` comes with completion definitions for its commands for bash and zsh. You
must manually configure your shell to enable completion support.

To make the completions available, source the definitions in your shell’s startup file
(e.g., `~/.bash_profile`, `~/.zshrc`, or `~/.profile`):

```bash
source $(mtt --shell-completion)
```

<!-- marker-issues -->

# Having problems or ideas?

Having a problem with metatrain? Please let us know by submitting an issue.

Submit new features or bug fixes through a pull request.

<!-- marker-contributing -->

# Contributors

Thanks goes to all people who make metatrain possible:

[![Contributors](https://contrib.rocks/image?repo=metatensor/metatrain)](https://github.com/metatensor/metatrain/graphs/contributors)

# Citing metatrain

If you found ``metatrain`` useful, you can cite its pre-print
(<https://doi.org/10.48550/arXiv.2508.15704>) as

```
@misc{metatrain,
title = {Metatensor and Metatomic: Foundational Libraries for Interoperable Atomistic
Machine Learning},
shorttitle = {Metatensor and Metatomic},
author = {Bigi, Filippo and Abbott, Joseph W. and Loche, Philip and Mazitov, Arslan
and Tisi, Davide and Langer, Marcel F. and Goscinski, Alexander and Pegolo, Paolo
and Chong, Sanggyu and Goswami, Rohit and Chorna, Sofiia and Kellner, Matthias and
Ceriotti, Michele and Fraux, Guillaume},
year = {2025},
month = aug,
publisher = {arXiv},
doi = {10.48550/arXiv.2508.15704},
}
```
