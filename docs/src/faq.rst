==========================
Frequently Asked Questions
==========================

Common questions from users learning to use metatrain. **New to machine learning?** Check
the :ref:`beginner_guide` first for core concepts.

If you don't find your answer here, please ask on our `Github discussion page`_!

.. _Github discussion page: https://github.com/metatensor/metatrain/discussions

Contents
--------

`Getting Started`_

`Training troubleshooting`_

`General training concepts`_

`Citing us`_

Getting Started
---------------
.. _Getting Started:

**Q: I'm completely new to machine learning. Where should I start?**

**A:** Welcome! Here's your path:

1. Read the :ref:`beginner_guide` to understand the concepts
2. Follow the :ref:`label_installation` instructions
3. Try the :ref:`label_quickstart` with a small example
4. Work through :ref:`sphx_glr_generated_examples_0-beginner_00-basic-usage.sh`

Don't worry about understanding everything at once - learning is iterative!

**Q: What data do I need to train a model?**

**A:** At minimum, you need atomic structures (positions, elements) with corresponding
energies. Ideally, also include forces. The data should be in a format ASE can read (like
XYZ files). See :ref:`sphx_glr_generated_examples_0-beginner_01-data_preparation.py` for
details on preparing your data.

**Q: How much data do I need?**

**A:** It depends on system complexity:

- **Simple systems** (small molecules, one element): 1,000-5,000 structures
- **Moderately complex** (multiple elements): 10,000-50,000 structures
- **Very complex** (many elements, phase transitions): 100,000+ structures

Start small to test your setup, then expand!

**Q: Which architecture should I choose as a beginner?**

**A:** Start with **SOAP-BPNN**:

- Easy to install and use
- Trains quickly (even on CPU)
- Good accuracy for many systems
- Fewer hyperparameters to worry about

Once comfortable, try **PET** for potentially better accuracy (requires GPU for practical
training).

Training troubleshooting
------------------------
.. _Training troubleshooting:

**Q: My training fails with an "out of memory" error, what can I do?**

**A:** This means your GPU or RAM doesn't have enough memory. Try these solutions in order:

1. **Reduce batch size**: In your options.yaml, add or decrease:

   .. code-block:: yaml

       training:
           batch_size: 8  # Start with 8, try smaller if needed

2. **Reduce cutoff radius**: Smaller cutoff = fewer neighbor interactions = less memory:

   .. code-block:: yaml

       architecture:
           model:
               cutoff: 4.0  # Try 4.0 instead of 5.0 or 6.0

3. **Use fewer features**: Check your architecture's documentation for options to reduce
   model size

4. **Use CPU**: If you have more RAM than GPU memory, try ``device: cpu`` (slower but more
   memory)

**Q: My training is very slow, what can I do?**

**A:** Several solutions to speed things up:

- **Use a GPU**: Add ``device: gpu`` to your options file. GPU training is often 10-100x
  faster than CPU, especially for PET.

- **Increase batch size**: Larger batches are more efficient (if you have enough memory):

  .. code-block:: yaml

      training:
          batch_size: 32  # Try larger values like 32, 64, 128

- **Use a faster architecture**: SOAP-BPNN trains much faster than PET, though with
  potentially lower accuracy

- **Reduce dataset size**: For initial testing, train on 1,000 structures before scaling
  up

- **Reduce cutoff**: Smaller cutoff radius means faster training (but may reduce accuracy)

**Q: My validation loss is not decreasing / training is not converging**

**A:** This suggests issues with your data or training setup:

1. **Check data quality**:

   - Are all calculations from the same method with same settings?
   - Plot energy-per-atom histogram - should be reasonable without strange spikes
   - Plot force distribution - very large forces (> 20 eV/Å) suggest problems
   - Look for outliers using parity plots (:ref:`sphx_glr_generated_examples_0-beginner_04-parity_plot.py`)

2. **Remove outliers**: Filter out structures with very large forces or unusual energies

3. **Check convergence**: Ensure your quantum calculations (DFT, etc.) were properly
   converged

4. **Try more epochs**: Sometimes models need longer to learn (50-200 epochs)

5. **Verify units**: Make sure energy and force units are correct in your options file

General training concepts
-------------------------
.. _General training concepts:

**Q: What cutoff radius should I use?**

**A:** The **cutoff radius** determines how far atoms "see" each other. Atoms beyond the
cutoff don't interact in the model.

**Rule of thumb:**

- **Molecules and small systems**: 4-6 Å works for most cases
- **Dense materials/liquids**: 6-8 Å to capture more neighbors
- **Long-range interactions**: 8-10 Å (but training will be slower)

**How to choose:**

1. Start with 5 Å - works for most systems
2. If accuracy is poor, try increasing to 6-7 Å
3. Test convergence: gradually increase cutoff and check if predictions improve

**Important notes:**

- Larger cutoff = more accurate but slower training and evaluation
- For architectures with **message passing** (like PET), the effective range is
  multiplied: 5 Å cutoff with 2 message-passing layers ≈ 10 Å effective range
- Chemical bonds are typically < 2 Å, so 5 Å captures first and second neighbors

**Example:** If you're modeling water, 5-6 Å captures the first hydration shell well.

**Q: In what format should I provide my data?**

**A:** The easiest format is **extended XYZ files** (``.xyz`` or ``.extxyz``), which ASE can
read and write.

Your XYZ file should contain:

- Atomic positions and elements (required)
- Energy in the ``info`` dictionary (required for energy training)
- Forces in the ``arrays`` dictionary (highly recommended)
- Cell information for periodic systems (if applicable)

**Simple example** of creating an XYZ file with ASE:

.. code-block:: python

    from ase.io import write

    # atoms is your ASE Atoms object
    atoms.info['energy'] = -100.5  # in eV
    atoms.arrays['forces'] = forces_array  # shape (n_atoms, 3)

    write('my_data.xyz', atoms, append=True)

See :ref:`sphx_glr_generated_examples_0-beginner_01-data_preparation.py` for more
details.

**Q: How accurate should my model be before using it for Molecular Dynamics?**

**A:** This depends on your system and goals. As a **general guideline**:

**Forces are most critical:**

- **Excellent**: < 1-2% error on force magnitudes
- **Good**: ~5% error (often acceptable for MD)
- **Marginal**: 10% error (usable but monitor carefully)
- **Poor**: > 10% error (likely unstable MD)

**Energy errors** are less critical for MD since we mostly care about forces (energy
gradients).

**Important**: Always test your MD simulations!

1. Start with short runs (1-10 ps)
2. Check for unphysical behavior:

   - Atoms flying apart (explosion)
   - Atoms collapsing (implosion)
   - Unreasonable temperatures or energies
   - Weird structural changes

3. If MD is unstable:

   - Add more diverse training data
   - Train longer
   - Check data quality
   - Consider a different architecture

The acceptable error also depends on:

- **Temperature**: Higher temperature MD is more forgiving
- **System**: Solids are easier than reactions
- **Purpose**: Screening vs. publication-quality simulations

**Q: How can I use a custom model architecture?**

**A:** You can add a new model architecture to metatrain, if you want to do so have a look at
:ref:`adding-new-architecture`. For adding a custom loss function have a look at :ref:`adding-new-loss`.

**Q: How can I visualize the results of my training?**

**A:** Every training run writes the train log into a csv file. You can use a simple python
script for e.g. parsing the losses.

**Q: How can I get uncertainties for my model?**

**A:** Have a look at the :ref:`LLPR tutorial <llprexample>`. It shows how to use models
with the last-layer prediction rigidity (`LLPR <LLPR_>`_) and local prediction rigidity (`LPR <LPR_>`_).

.. _LLPR: https://arxiv.org/html/2403.02251v1
.. _LPR: https://pubs.acs.org/doi/10.1021/acs.jctc.3c00704

**Q: How can I save and restart my training?**

**A:** Metatrain offers a convenient and automatic way to restart models from checkpoints.
Please have a look at :ref:`checkpoints` for details.

Citing us
---------
.. _Citing us:

**Q: How do I cite metatrain?**

**A:** Please follow the instructions on :ref:`this page <citingmetatrain>`.

