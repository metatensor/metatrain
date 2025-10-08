Training a model from scratch
#############################
This tutorial explains how to train a model with ``metatrain`` from scratch and evaluate
it. `This dataset <../../../../examples/ase/ethanol_reduced_100.xyz>`_ is used here as an example of the preferred dataset format. If you
have your own dataset, you can simply replace the dataset file name with yours.

Train the model
---------------

Configure the ``options.yaml`` and run the training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Below is an example ``options.yaml`` for training a PET model. In order to train other
models, simply replace the architecture name with other models' architecture name. For
the supported models, please check `Available Architectures`_ .

.. _`Available Architectures`: https://metatensor.github.io/metatrain/latest/architectures/index.html

.. code-block:: yaml

    architecture:
      name: pet  # the architecture name for PET, or the name for other architectures,
      if you want to train other models

      model:
        cutoff: 4.5  # the cutoff

      training:
        num_epochs: 10  # this is for a reasonable time of a tutorial, for a good model, consider increasing the number
        batch_size: 10  # the size of the training data feed to the model per batch, determining the GPU memory usage during the training
        log_interval: 1
        checkpoint_interval: 10  # it saves checkpoints of the model every 10 epochs


    # this needs specifying based on the specific dataset
    training_set:
      systems:
        read_from: ../../../../examples/ase/ethanol_reduced_100.xyz
        length_unit: Angstrom
      targets:
        energy:
          key: energy # name of the target value
          unit: eV # unit of the target value

    test_set: 0.1 # 10 % of the training_set are randomly split and taken for test set
    validation_set: 0.1 # 10 % of the training_set are randomly split and for validation set

Copy-pasting this content into ``options.yaml``, and run

.. code-block:: bash

    mtt train options.yaml

It will start training. ``metatrain`` will automatically read the atomic forces from the training set, if they are stored in it and named as "forces". The model can also be trained to learn other properties through transfer learning. For this, please refer to this `transfer learning tutorial`_.

.. _`transfer learning tutorial`: https://metatensor.github.io/metatrain/latest/advanced-concepts/transfer-learning.html

Once the training is started, a folder named ``outputs`` will be created automatically under the folder where you run the command. Under this ``outputs`` folder, there is a folder with the timestamp. Below is a normal structure of that folder of a successful training run.

.. code-block:: bash

    outputs/2025-10-07/17-08-25/
    ├── indices  # the results of dataset-spliting
    │   ├── test.txt
    │   ├── training.txt
    │   └── validation.txt
    ├── model_0.ckpt  # the intermediate model saved at the 0th training step
    ├── model.ckpt  # the final model
    ├── model.pt  # the final model in .pt format, can be directly used by ASE and LAMMPS
    ├── options_restart.yaml  # an expanded options file
    ├── train.csv  # a structured output of training criteria, like training losses, energy MAEs, and force RMSEs
    └── train.log  # a human-friendly output

The ``train.log`` provides information of the training procedure. For example, by checking the following

.. code-block:: bash

    [2025-10-07 17:08:25][INFO] - Setting up training set
    [2025-10-07 17:08:25][INFO] - Forces found in section 'energy', we will use this gradient to train the model
    [2025-10-07 17:08:25][WARNING] - No stress found in section 'energy'.

you can know the forces are identified by ``metatrain`` and are used during the training, and it fails to find stress. The following provides some statistical of the training, validation, and the test set

.. code-block:: bash

    [2025-10-07 17:08:25][INFO] - Training dataset:
        Dataset containing 80 structures
        Mean and standard deviation of targets:
        - energy:
          - mean -9.708e+04 eV
          - std  3.97 eV
    [2025-10-07 17:08:25][INFO] - Validation dataset:
        Dataset containing 10 structures
        Mean and standard deviation of targets:
        - energy:
          - mean -9.708e+04 eV
          - std  3.73 eV
    [2025-10-07 17:08:25][INFO] - Test dataset:
        Dataset containing 10 structures
        Mean and standard deviation of targets:
        - energy:
          - mean -9.708e+04 eV
          - std  3.535 eV

The training metrics are outputted every epoch, like

.. code-block:: bash

    [2025-10-07 17:08:28][INFO] - Epoch:    0 | learning rate: 0.000e+00 | training loss: 6.305e+03 | training energy RMSE (per atom): 884.08 meV | training energy MAE (per atom): 773.44 meV | training forces RMSE: 28059.9 meV/A | training forces MAE: 20581.1 meV/A | validation loss: 7.725e+02 | validation energy RMSE (per atom): 877.08 meV | validation energy MAE (per atom): 772.04 meV | validation forces RMSE: 27779.2 meV/A | validation forces MAE: 20201.9 meV/A

These metrics are also outputted into ``train.csv`` in a formatted way, which can be used for plotting graph like loss curve.

It is easy to restart the training from the last step, by running

.. code-block:: bash

    mtt train options.yaml --restart model.ckpt

Evaluate the trained model
--------------------------
In order to evaluate the model on the test set, we can use the mtt eval sub-command. First, create the input file ``eval.yaml`` with the following options:

.. code-block:: yaml

    systems:
      read_from: ../../../../examples/ase/ethanol_reduced_100.xyz # file where the positions are stored
      length_unit: Angstrom
    targets:
      energy:
        key: energy # name of the target value
        unit: eV # unit of the target value

and run

.. code-block:: bash

    mtt eval PATH_TO_YOUR_MODEL/model.pt eval.yaml  # be sure to replace the path

After this, a file named ``output.xyz`` will be created, with the atom positions and the predicted forces recorded in it. Also, you should see these statistical on your screen

.. code-block:: bash

    [2025-10-07 17:11:47][INFO] - energy RMSE (per atom): 436.50 meV | energy MAE (per atom): 341.32 meV | forces RMSE: 27823.1 meV/A | forces MAE: 20392.7 meV/A
    [2025-10-07 17:11:47][INFO] - Evaluation time: 1.10 s [1.2185 ± 1.2768 ms per atom]

Further analysis can be performed now that the model is trained. We provide a `Python script`_ that can be used to generate a parity plot of the target vs predicted energies, but otherwise leave this open-ended.

.. _`Python script`: https://raw.githubusercontent.com/metatensor/Workshop-spring-2025/refs/heads/main/training-custom-models/part-1-gap/parity_plot.py

To run the script, download it from the repository, modify the paths as necessary (indicated with a #TODO), and run. This will generate a plot saved at parity_plot.png.


Use the model
-------------------------
With the trained model, you can run molecular dynamics. Please refer to these two tutorials for `ASE`_ and `LAMMPS`_ to see how to do that.

.. _`ASE`: https://docs.metatensor.org/metatomic/latest/examples/2-running-ase-md.html

.. _`LAMMPS`: https://atomistic-cookbook.org/examples/pet-mad-nc/pet-mad-nc.html#running-lammps-on-gpus-with-kokkos
