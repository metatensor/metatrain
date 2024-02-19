metatensor-models
-----------------

|tests| |docs|

.. warning::

    **metatensor-models is still very early in the concept stage. You should not use it
    for anything important.**

This is a repository for models using metatensor, in one shape or another. The only
requirement is for these models to be able to take metatensor objects as inputs and
outputs. The models do not need to live entirely in this repository: in the most extreme
case, this repository can simply contain a wrapper to an external model.

.. marker-introduction

What is metatensor-models?
###########################

The idea behind ``metatensor-models`` is to have a general hub
that provide an homogeneous enviroment and user interface to
train, export and evaluate ML models and to
connect those models with various MD engines (e.g. ``LAMMPS``, ``i-PI``, ``ASE`` ...).
``metatensor-models`` is the tool that transforms every ML architecture in an
end-to-end model. Any custom ML architecture compatible with TorchScript
can be integrated in ``metatensor-models``, gaining automatic access to
a training and evaluation interface, as well as compatibility with various MD engines.

Note: ``metatensor-models`` does not provide `per se` mathematical functionalities but
relies on external models that implement the various architectures.

Features
#########

- **Custom ML Architecture**: Integrate any TorchScriptable ML model
  to explore innovative architectures.

- **MD Engine Compatibility**: Supports various MD engines for diverse research
  and application needs.

- **Streamlined Training**: Automated process leveraging MD-generated data to
  optimize ML models with minimal effort. It uses the ``hydra`` module to
  easy management of folder and files.

- **HPC Compatibility**: Efficient in HPC environments for extensive simulations.

- **Future-Proof**: Extensible to accommodate advancements in ML and MD fields.

.. marker-architectures

List of Implemented Architectures
###################################

Currently metatensor-models supports the following architectures for building an
atomistic model.

.. list-table::
   :widths: 34 66
   :header-rows: 1

   * - Name
     - Description

   * - SOAP BPNN
     - A Behler-Parrinello neural network with SOAP features

.. marker-documentation

Documentation
------------

For details, tutorials, and examples, please have a look at our
`documentation <https://lab-cosmo.github.io/metatensor-models/latest/>`_.

.. marker-installation

Installation
------------

You can install metatensor-models with pip:

.. code-block:: bash

    git clone https://github.com/lab-cosmo/metatensor-models
    cd metatensor-models
    pip install .


In addition, specific models must be installed by specifying the model name. For
example, to install the SOAP-BPNN model, you can run:

.. code-block:: bash

    pip install .[soap-bpnn]

.. marker-issues

Having problems or ideas?
-------------------------

Having a problem with metatensor-models? Please let us know by `submitting an issue
<https://github.com/metatensor-models/issues>`_.

Submit new features or bug fixes through a `pull request
<https://github.com/metatensor-models/pulls>`_.

.. marker-contributing

Contributors
------------

Thanks goes to all people that make metatensor-models possible:

.. image:: https://contrib.rocks/image?repo=lab-cosmo/metatensor-models
   :target: https://github.com/lab-cosmo/metatensor-models/graphs/contributors

.. |tests| image:: https://github.com/lab-cosmo/metatensor-models/workflows/Tests/badge.svg
   :alt: Github Actions Tests Job Status
   :target: (https://github.com/lab-cosmo/metatensor-models/\
                actions?query=workflow%3ATests)

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
   :alt: Python
   :target: https://lab-cosmo.github.io/metatensor-models/latest/
