metatrain
---------

|tests| |codecov| |docs|

.. warning::
  **metatrain is still very early in the concept stage. You should not use it
  for anything important.**

This is a repository for training and evaluating machine learning models from various
architectures for atomistic systems. The only requirement is for an architecture to
be able to take metatensor_ objects as inputs/outputs and have to be JIT compilable
using TorchScript_. The architectures do not need to live entirely in this repository:
in the most extreme case, this repository can simply contain a wrapper to an external
architecture.

.. marker-introduction

What is metatrain?
##################
``metatrain`` is a command line interface (cli) to ``train`` and ``evaluate`` atomistic
models of various architectures. It features a common ``yaml`` option inputs to
configure training and evaluation. Trained models are exported as standalone files that
can be used directly in various molecular dynamics (MD) engines (e.g. ``LAMMPS``,
``i-PI``, ``ASE`` ...) using the metatensor_ atomistic interface.

The idea behind ``metatrain`` is to have a general hub that provide an homogeneous
environment and user interface transforms every ML architecture in an end-to-end model
that can be connected to an MD engine. Any custom architecture compatible with
TorchScript_ can be integrated in ``metatrain``, gaining automatic access to a training
and evaluation interface, as well as compatibility with various MD engines.

Note: ``metatrain`` does not provide mathematical functionalities *per se* but relies on
external models that implement the various architectures.

.. _TorchScript: https://pytorch.org/docs/stable/jit.html
.. _metatensor: https://docs.metatensor.org

Features
########
- **Custom ML Architecture**: Integrate any TorchScriptable ML model
- **MD Engine Compatibility**: Supports various MD engines for diverse research and
  application needs.
- **Streamlined Training**: Automated process leveraging MD-generated data to optimize
  ML models with minimal effort.
- **HPC Compatibility**: Efficient in HPC environments for extensive simulations.
- **Future-Proof**: Extensible to accommodate advancements in ML and MD fields.

.. marker-architectures

List of Implemented Architectures
#################################
Currently ``metatrain`` supports the following architectures for building an atomistic
model.

.. list-table::
  :widths: 34 66
  :header-rows: 1

  * - Name
    - Description
  * - GAP
    - Sparse Gaussian Approximation Potential (GAP) using Smooth Overlap of Atomic
      Positions (SOAP).
  * - SOAP BPNN
    - A Behler-Parrinello neural network with SOAP features
  * - Alchemical Model
    - A Behler-Parrinello neural network with SOAP features and Alchemical Compression
      of the composition space
  * - PET
    - Point Edge Transformer (PET), interatomic machine learning potential

.. marker-documentation

Documentation
-------------
For details, tutorials, and examples, please have a look at our
`documentation <https://metatensor.github.io/metatrain/latest/>`_.

.. marker-installation

Installation
------------
You can install ``metatrain`` with pip:

.. code-block:: bash

    git clone https://github.com/lab-cosmo/metatrain
    cd metatrain
    pip install .

In addition, specific models must be installed by specifying the model name. For
example, to install the SOAP-BPNN model, you can run:

.. code-block:: bash

    pip install .[soap-bpnn]

You can then use ``mtt`` from the command line to train your models!

Shell Completion
################
``metatrain`` comes with completion definitions for its commands for ``bash`` and
``zsh``. Since it is difficult to automatically configure shell completions in a robust
manner, you must manually configure your shell to enable its completion support.

To make the completions available, source the definitions as part of your shell's
startup. Add the following to your ``~/.bash_profile``, ``~/.zshrc`` (or, if they don't
exist, ``~/.profile``):

.. code-block:: bash

  source $(mtt --shell-completion)

.. marker-issues

Having problems or ideas?
-------------------------
Having a problem with metatrain? Please let us know by `submitting an issue
<https://github.com/metatensor/metatrain/issues>`_.

Submit new features or bug fixes through a `pull request
<https://github.com/metatensor/metatrain/pulls>`_.

.. marker-contributing

Contributors
------------
Thanks goes to all people that make ``metatrain`` possible:

.. image:: https://contrib.rocks/image?repo=lab-cosmo/metatrain
  :target: https://github.com/metatensor/metatrain/graphs/contributors

.. |tests| image:: https://github.com/lab-cosmo/metatrain/workflows/Tests/badge.svg
  :alt: Github Actions Tests Job Status
  :target: https://github.com/metatensor/metatrain/actions?query=branch%3Amain

.. |codecov| image:: https://codecov.io/gh/lab-cosmo/metatrain/branch/main/graph/badge.svg
  :alt: Code coverage
  :target: https://codecov.io/gh/metatensor/metatrain

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
  :alt: Documentation
  :target: https://metatensor.github.io/metatrain/latest
