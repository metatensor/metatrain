metatrain
=========

.. image:: https://raw.githubusercontent.com/metatensor/metatrain/refs/heads/main/docs/src/logo/metatrain.svg
   :width: 200 px
   :align: left

|tests| |codecov| |docs|

.. marker-introduction

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

.. marker-architectures

List of Implemented Architectures
---------------------------------

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
  * - PET
    - Point Edge Transformer (PET), interatomic machine learning potential
  * - NanoPET (*experimental*)
    - re-implementation of the original PET with slightly improved training and
      evaluation speed
  * - NativePET (*experimental*)
    - re-implementation of the original PET, preserving the original architecture
      and providing a clean code implementation and additional features
  * - SOAP BPNN
    - A Behler-Parrinello neural network with SOAP features

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

    pip install metatrain

In addition, specific models must be installed by specifying the model name. For
example, to install the *SOAP-BPNN* model, you can run:

.. code-block:: bash

    pip install metatrain[soap-bpnn]

You can then use ``mtt`` from the command line to train your models!

.. marker-quickstart

Quickstart
----------

To train a model, you can use the following command:

.. code-block:: bash

    mtt train options.yaml

Where ``options.yaml`` is a configuration file that specifies the training options. For
example, the following configuration file trains a *SOAP-BPNN* model on the QM9 dataset:

.. code-block:: yaml

    # architecture used to train the model
    architecture:
      name: soap_bpnn
      training:
        num_epochs: 5 # a very short training run

    # Mandatory section defining the parameters for system and target data of the
    # training set
    training_set:
      systems: "qm9_reduced_100.xyz" # file where the positions are stored
      targets:
        energy:
          key: "U0" # name of the target value
          unit: "eV" # unit of the target value

    test_set: 0.1 # 10 % of the training_set are randomly split and taken for test set
    validation_set: 0.1 # 10 % of the training_set are randomly split and for validation set

.. marker-shell

Shell Completion
----------------

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

.. image:: https://contrib.rocks/image?repo=metatensor/metatrain
  :target: https://github.com/metatensor/metatrain/graphs/contributors

.. |tests| image:: https://img.shields.io/github/checks-status/metatensor/metatrain/main
  :alt: Github Actions Tests Job Status
  :target: https://github.com/metatensor/metatrain/actions?query=branch%3Amain

.. |codecov| image:: https://codecov.io/gh/metatensor/metatrain/branch/main/graph/badge.svg
  :alt: Code coverage
  :target: https://codecov.io/gh/metatensor/metatrain

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
  :alt: Documentation
  :target: https://metatensor.github.io/metatrain/latest
