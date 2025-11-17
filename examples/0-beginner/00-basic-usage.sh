# .. _label_basic_usage:
#
# Basic Usage
# ===========
#
# This tutorial shows you how to train and evaluate a machine learning model using
# ``metatrain``. **No Python coding required** - everything is done from the command line!
#
# ``metatrain`` is accessed through the ``mtt`` command. Let's start by checking that
# everything is installed correctly:
#

mtt --help

# %%
#
# Training Your First Model
# --------------------------
#
# Now we'll train a machine learning model to predict molecular energies. We'll use:
#
# - **Architecture**: SOAP-BPNN (a good beginner-friendly choice)
# - **Dataset**: A subset of the QM9 dataset with 100 molecules
#
# You can download the dataset here: :download:`qm9_reduced_100.xyz <qm9_reduced_100.xyz>`
#
# What is the QM9 dataset?
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# QM9 is a well-known dataset of small organic molecules (up to 9 heavy atoms) with
# accurate quantum chemistry calculations. It's great for learning because:
#
# - Small molecules train quickly
# - Well-studied, so we know what to expect
# - Diverse enough to demonstrate model capabilities
#
# Reference: `Ramakrishnan et al., Scientific Data 2014
# <https://www.nature.com/articles/sdata201422>`_
#
# Creating the Configuration File
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Training is controlled by a YAML configuration file. Here's a minimal example that uses
# default settings for most parameters:
#
# .. literalinclude:: options-basic.yaml
#    :language: yaml
#
# **What's in this file?**
#
# - ``architecture: name: soap_bpnn`` - Use the SOAP-BPNN neural network
# - ``training: num_epochs: 5`` - Train for 5 epochs (very short, just for demo)
# - ``training_set`` - Points to our data file and tells metatrain we're predicting U0
#   energy in eV units
# - ``test_set: 0.1`` - Hold out 10% of data for final testing
# - ``validation_set: 0.1`` - Hold out 10% to monitor training progress
#
# Starting the Training
# ^^^^^^^^^^^^^^^^^^^^^
#
# Create the ``options-basic.yaml`` file in your current directory and run:


mtt train options-basic.yaml

# %%
#
# What Happens During Training?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As the model trains, you'll see:
#
# - **Loss values**: Should generally decrease (means the model is learning)
# - **RMSE/MAE**: Error metrics for energy and force predictions
# - **Epoch numbers**: Progress through the training data (1 epoch = seeing all data once)
#
# The training creates several outputs:
#
# - ``outputs/YYYY-MM-DD/HH-MM-SS/`` - A timestamped folder with all results
# - ``model.pt`` - Your trained model (this is what you'll use for predictions!)
# - ``model.ckpt`` - A checkpoint file (for resuming training if needed)
# - ``train.log`` - Human-readable log of what happened
# - ``train.csv`` - Structured data for plotting training curves
# - ``options_restart.yaml`` - Complete configuration with all defaults expanded
# - ``extensions/`` - Compiled code needed by some architectures (if applicable)
#
# For more training options, run:
#

mtt train --help

# %%
#
# After training finishes, you'll have ``model.pt`` and ``model.ckpt`` files both in your
# current directory and in the timestamped output folder.
#
# Evaluating Your Model
# ---------------------
#
# Now let's test how well our trained model performs! Evaluation means using the model to
# make predictions on structures and comparing them to the true values (if available).
#
# Creating the Evaluation Configuration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create an ``eval-basic.yaml`` file. This is simpler than the training config:
#
# .. literalinclude:: eval-basic.yaml
#    :language: yaml
#
# **What's in this file?**
#
# - ``systems`` - The structures you want to predict for
# - ``targets`` - (Optional) The true values to compare against. If included, metatrain
#   will calculate error metrics (RMSE, MAE)
#
# If you omit ``targets``, the model will just make predictions without error reporting.
#
# Running the Evaluation
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Evaluate the model on our dataset. The first argument is the trained model, the second
# is the evaluation configuration. The ``-e extensions/`` flag includes any compiled model
# extensions if needed:

mtt eval model.pt eval-basic.yaml -e extensions/

# %%
#
# Understanding the Evaluation Output
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The evaluation does two things:
#
# 1. **Prints error metrics** to the terminal (RMSE, MAE) comparing predictions to true
#    values
# 2. **Creates output.xyz** - a file with structures and predicted properties
#
# Let's look at the beginning of the output file:

head -n 20 output.xyz

# %%
#
# The ``output.xyz`` file contains all the input structures with added predicted
# properties. You can use this with visualization tools or for further analysis.
#
# More Evaluation Options
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# See all available evaluation options:

mtt eval --help

# %%
#
# **Useful options:**
#
# - ``-b`` or ``--batch-size``: Process multiple structures at once (faster for large
#   datasets)
# - ``-o`` or ``--output``: Specify a different output filename
#
# Next Steps
# ----------
#
# Congratulations! You've trained and evaluated your first atomistic machine learning
# model. Here's what you can do next:
#
# 1. **Run molecular dynamics**: Use your model in simulations. See
#    :ref:`sphx_glr_generated_examples_0-beginner_05-run_ase.py` for how to run MD with
#    ASE.
#
# 2. **Analyze your model**: Create parity plots to visualize prediction quality. See
#    :ref:`sphx_glr_generated_examples_0-beginner_04-parity_plot.py`.
#
# 3. **Train on your own data**: Prepare your structures and energies in XYZ format
#    (see :ref:`sphx_glr_generated_examples_0-beginner_01-data_preparation.py`) and follow
#    this same process.
#
# 4. **Improve your model**: Try different architectures, adjust hyperparameters, or add
#    more training data.
#
# 5. **Learn advanced features**: Explore :ref:`Fine-tuning <sphx_glr_generated_examples_0-beginner_02-fine-tuning.py>`
#    or other tutorials in the documentation.
