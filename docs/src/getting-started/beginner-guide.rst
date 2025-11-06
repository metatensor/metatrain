.. _beginner_guide:

Beginner's Guide to Machine Learning for Atomistic Systems
==========================================================

This guide explains the key concepts you need to understand when training machine
learning models for atomistic systems. **No prior machine learning experience is
required!**

What is Machine Learning for Atoms and Molecules?
-------------------------------------------------

Traditional quantum mechanical calculations (like Density Functional Theory or DFT) can
accurately calculate properties of molecules and materials, but they're very slow. Machine
learning offers a solution: we can train a model to learn from these accurate calculations
and then make predictions much faster—often thousands of times faster!

**The basic idea:**

1. **Training data**: You provide examples of atomic structures and their properties
   (energies, forces, etc.) calculated with accurate methods
2. **Learning**: The machine learning model finds patterns in how atomic arrangements
   relate to properties
3. **Prediction**: Once trained, the model can predict properties for new structures
   it hasn't seen before

Think of it like teaching someone to estimate house prices: show them many examples of
houses with their prices, and they learn to predict prices for new houses based on
features like size, location, and age.

Key Concepts Explained
----------------------

Training, Validation, and Test Sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When training a machine learning model, we split our data into three parts:

- **Training set** (typically 70-80% of data): The model learns from these examples. This
  is like giving a student practice problems.

- **Validation set** (typically 10-15% of data): Used during training to check if the
  model is learning well or overfitting. This is like giving a student quiz problems to
  check their understanding.

- **Test set** (typically 10-15% of data): Used **only at the end** to evaluate final
  performance. This is like the final exam - the student hasn't seen these problems during
  study.

**Why split the data?** If we only used a training set, the model might just "memorize"
the examples rather than learning general patterns. The validation and test sets help us
verify that the model has learned generalizable knowledge.

Epochs and Training Iterations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Epoch**: One complete pass through all training data. If you have 1000 structures and
  train for 100 epochs, the model sees each structure 100 times.

- **Batch size**: How many structures the model processes at once. For example, with 1000
  structures and a batch size of 32, each epoch involves ~31 steps (batches).

**How many epochs do I need?** Typically 50-200 epochs for most systems. Too few, and the
model hasn't learned enough. Too many, and it might "overfit" (memorize training data
instead of learning patterns).

Overfitting vs Underfitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Underfitting**: The model is too simple and hasn't learned enough. Like trying to fit
  a curve through complex data with just a straight line.

  *Signs*: High errors on both training and validation sets.

  *Solution*: Train longer (more epochs) or use a more complex model.

- **Overfitting**: The model has memorized the training data but doesn't generalize. Like
  memorizing answers to practice problems without understanding the concepts.

  *Signs*: Low training error but high validation error.

  *Solution*: Train for fewer epochs, use more training data, or simplify the model.

Loss Functions
^^^^^^^^^^^^^^

The **loss function** measures how wrong the model's predictions are. During training, the
model adjusts itself to minimize this loss.

For atomistic systems, common losses include:

- **Energy loss**: Difference between predicted and true energies
- **Force loss**: Difference between predicted and true forces (derivatives of energy)
- **Stress loss**: Difference between predicted and true stress (for periodic systems)

You can train on multiple properties simultaneously by combining their losses with
different weights.

Learning Rate
^^^^^^^^^^^^^

The **learning rate** controls how much the model adjusts itself based on errors. Think of
it like step size when searching for the bottom of a valley:

- **Too large**: The model jumps around and might miss the best solution
- **Too small**: Training is very slow
- **Just right**: The model efficiently finds a good solution

Most architectures in metatrain use adaptive learning rates that automatically adjust
during training, so you usually don't need to worry about this.

Model Architectures
-------------------

An **architecture** is the specific type of machine learning model you're using. Different
architectures have different strengths:

SOAP-BPNN (Good for beginners)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What it is**: A neural network that uses SOAP (Smooth Overlap of Atomic Positions)
  descriptors to represent atomic environments
- **Pros**: Fast to train, works well for many systems, good accuracy
- **Cons**: Less accurate than more modern methods for some systems
- **Best for**: Getting started, small to medium datasets, systems without long-range
  interactions

PET (Point Edge Transformer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What it is**: A modern graph neural network using message passing
- **Pros**: State-of-the-art accuracy, handles complex systems well
- **Cons**: Slower to train, requires more data, works best with GPU
- **Best for**: Systems requiring high accuracy, larger datasets, users with GPU access

GAP (Gaussian Approximation Potential)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **What it is**: A kernel-based method using SOAP descriptors
- **Pros**: Very accurate, provides uncertainty estimates
- **Cons**: Training can be slow for large datasets
- **Best for**: Smaller datasets (< 5000 structures), when uncertainty quantification is
  important

Understanding Your Data
-----------------------

Data Format
^^^^^^^^^^^

Metatrain works with atomic structures that include:

- **Positions**: 3D coordinates of each atom
- **Elements**: Chemical element of each atom (e.g., H, C, O)
- **Cell**: For periodic systems, the simulation box dimensions
- **Properties**: Target values you want to predict (energy, forces, etc.)

The most common format is **XYZ files** (specifically extended XYZ), which are text files
that ASE can read and write. Each structure in the file includes atomic positions and
properties.

Data Quality
^^^^^^^^^^^^

**Good training data is crucial!** Your model can only be as good as your training data.

Essential requirements:

- **Consistency**: All data calculated with the same method and settings
- **Convergence**: Quantum mechanical calculations must be well-converged
- **Representative**: Cover the range of structures you want to predict
- **Clean**: Remove or fix obvious errors and outliers

**How much data do I need?** This depends on system complexity:

- **Simple systems** (small molecules, single element): 1,000-5,000 structures
- **Complex systems** (multiple elements, reactions): 10,000-100,000 structures
- **Very complex systems** (many elements, phase transitions): 100,000+ structures

Data Preprocessing
^^^^^^^^^^^^^^^^^^

Consider these preprocessing steps:

1. **Remove outliers**: Structures with unusually large forces or energies (often
   unphysical or poorly converged calculations)

2. **Check energy distribution**: Plot energy per atom histogram - should be reasonable
   and without strange spikes

3. **Force analysis**: Very large forces (> 20 eV/Å) often indicate problems and can make
   training difficult

4. **Balance your dataset**: Include diverse configurations (different structures,
   temperatures, deformations) not just similar ones

Practical Training Tips
-----------------------

Starting Your First Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Start simple**: Use default hyperparameters for your chosen architecture
2. **Use a small dataset first**: Test with 100-1000 structures to verify everything works
3. **Short initial run**: Try 5-10 epochs first to check for obvious problems
4. **Monitor progress**: Watch the validation loss - it should decrease

Monitoring Training
^^^^^^^^^^^^^^^^^^^

During training, metatrain outputs metrics that tell you how well training is going:

- **Loss**: Should generally decrease over epochs (some fluctuation is normal)
- **RMSE (Root Mean Square Error)**: Average prediction error
- **MAE (Mean Absolute Error)**: Another measure of prediction error

**What to look for:**

- Validation loss decreasing: ✓ Good! The model is learning
- Validation loss increasing: ✗ Stop training, you're overfitting
- Validation loss flat: → Model has finished learning or needs different hyperparameters

Common Problems and Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem: "Out of memory" error**

- Reduce batch size in your options file
- Use fewer structures
- Reduce model size (e.g., smaller cutoff radius)

**Problem: Training is very slow**

- Use a GPU if available (add ``device: cuda`` to options)
- Increase batch size
- Consider a faster architecture (e.g., SOAP-BPNN instead of PET)

**Problem: Poor accuracy**

- Check your data quality (see above)
- Train for more epochs
- Use more training data
- Try a different architecture or adjust hyperparameters

**Problem: Model works on validation but fails in MD simulations**

- Your training data may not be representative enough
- Add more diverse structures (different temperatures, deformations)
- Check for energy/force units consistency

Next Steps
----------

Now that you understand the basics, you're ready to:

1. :ref:`Install metatrain <label_installation>`
2. Follow the :ref:`Quickstart tutorial <label_quickstart>`
3. Try the :ref:`Basic usage example <label_basic_usage>`
4. Explore :ref:`Training from scratch <sphx_glr_generated_examples_0-beginner_03-train_from_scratch.py>`

Remember: machine learning is iterative! Your first model might not be perfect, and that's
okay. Each training run teaches you something about your system and how to improve.

Glossary of Terms
-----------------

- **Architecture**: The type of machine learning model (e.g., SOAP-BPNN, PET, GAP)
- **Batch size**: Number of structures processed together in one step
- **Checkpoint**: A saved state of the model during training (for resuming later)
- **Cutoff radius**: Maximum distance for considering atomic interactions
- **Descriptor**: Mathematical representation of atomic environments
- **DFT**: Density Functional Theory - a quantum mechanical method for calculating properties
- **Epoch**: One complete pass through the training data
- **Hyperparameter**: Settings that control training (learning rate, batch size, etc.)
- **Loss function**: Measure of prediction error that the model tries to minimize
- **Overfitting**: When a model memorizes training data instead of learning patterns
- **RMSE**: Root Mean Square Error - a measure of prediction accuracy
- **Underfitting**: When a model hasn't learned enough from the data
- **Validation set**: Data used during training to monitor generalization
