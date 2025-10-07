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

**A:** 

**Q:** My training is very slow, what can I do? \

**A:**

**Q:**  My training is not converging, what can I do? \

**A:**

General training concepts
-------------------------
.. _General training concepts:

**Q:** What cutoff radius should I use? \

**A:** The optimal cutoff radius depends on the type of system you are modeling.

In general, the cutoff should be large enough to include all physically relevant interactions (e.g., chemical bonding and short-range correlations) but not so large that it adds unnecessary computational cost. For istance around **4–6 Å** is a good value for most systems, but it can be increased to **8-10 Å** for condensed phases where longer-range effects are important. You can then test convergence by gradually increasing the cutoff and monitoring whether your target quantities (energies, forces, or other observables) change significantly.

Note that if you are using massage passing the effective cutoff extends beyond the nominal atomic cutoff. For example, a model with a 5 Å cutoff and two message-passing layers (PET default hypers) can capture correlations up to roughly 10 Å.

**Q:** In what format should I provide my data? \

**A:**

**Q:** How good should my errors are before I can use my model to run Molecular Dynamics simulations? \

**A:**

**Q:** How can I use a custom model architecture? \

**A:**

**Q:** How can I visualize the results of my training? \

**A:**

Citing us
---------
.. _Citing us:

**Q:** How do I cite ``metatrain``?

**A:** Please follow the instructions on :ref:`this page <citingmetatrain>`.

