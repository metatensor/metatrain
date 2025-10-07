==========================
Frequently Asked Questions
==========================

This page is a collection of FAQs. Below are links to the sections:

Contents
--------


`Training troubleshooting`_\

`General training concepts`_ \

`Citing us`_ \

Training troubleshooting
------------------------
.. _Training troubleshooting:

**Q:** My training fails with an out of memory error, what can I do? \

**A:** Reduce batch size. 

**Q:** My training is very slow, what can I do? \

**A:** There are several reasons why training can be slow. If possible, 
try to reduce the dataset size or increase the batch size. 
You can also try to run on a GPU, which significantly increases performance times.

**Q:**  My training is not converging, what can I do? \

**A:** Please make sure that you dataset is consitently computed and converged to a reasonable accuracy.

General training concepts
-------------------------
.. _General training concepts:

**Q:** What cutoff radius should I use? \

**A:**

**Q:** In what format should I provide my data? \

**A:**

**Q:** How good should my errors are before I can use my model to run Molecular Dynamics simulations? \

**A:**

**Q:** How can I use a custom model architecture? \

**A:** You can add a new model architecture to metatrain, if you want to do so have a look at
`adding-new-architecture`. For adding a custom loss function have a look at `adding-new-loss`.
If you just want to change the hyperparameters of an existing model architecture when training
have a look at `train_yaml_config`.

**Q:** How can I visualize the results of my training? \

**A:** Every training run writes the train log into a csv file. You can use a simple python 
script for e.g. parsing the losses. A small example is shown in 
:ref:`visualize-training <visualize-training>`. For other examples, check out examples in the 
`AtomisticCookbook <https://atomistic-cookbook.org/examples/pet-finetuning/pet-ft.html>`. 
For tracking your training runs live, there is also the possibility to connect to wandb. 
For seeing how to link the wandb logger, follow the section 
:ref:`Advanced Base configuration <_advanced_base_conf>`.

**Q:** How can I get uncertainties for my model? \

**A:** 

**Q:** How can save and restart my training? \

**A:** Metatrain offers a convenient and automatic way to restart models from checkpoints.
Please have a look at `` for details.

Citing us
---------
.. _Citing us:

**Q:** How do I cite ``metatrain``?

**A:** Please follow the instructions on :ref:`this page <citingmetatrain>`.

