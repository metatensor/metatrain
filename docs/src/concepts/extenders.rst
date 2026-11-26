.. _concept-extenders:

Using extenders
===============

Extenders are a powerful way to extend the functionality of architectures in ``metatrain``.

For a list of available extenders and their functionality, see :ref:`available-extenders`.

.. warning::

    Extenders are a big infrastructural change that has been recently added
    to ``metatrain``. As such, it is possible that it contains bugs or that
    the usage changes in the future. If you find any issues or have suggestions,
    feedback is highly appreciated.

What extenders do
-----------------

They implement functions that are called in the forward pass, after the core of the
model. Their purpose is to augment the capabilities of all architectures in a modular
way, so that the functionality is immediately available to all architectures without
having to modify their code. Any new architecture that is added to ``metatrain``
will automatically benefit from the extenders.

Typical examples of extenders are:

- Readout layers that implement different pooling strategies (e.g.
  :ref:`the intensive gap extender <extender-intensive_gap>` implements a pooling strategy that is
  invariant to the number of atoms in the system).
- Modules that produce outputs that the architecture does not natively support
  (e.g. :ref:`the tensor basis extender <extender-tensor_basis>` produces spherical outputs
  from scalar outputs of the architecture).


Usually, the way extenders work is that you indicate the target of your dataset that
the extender should produce, and then the extender will automatically request the necessary
intermediate outputs from the architecture. However, one can do more complex stuff
like chaining extenders or specifying the outputs of the architecture that the extender
should use as input. The exact functionality really depends on each particular extender.

By default, the outputs that an extender requests from the architecture are named
``mtt::extender_aux::<extender_name>::...``, for example
``mtt::extender_aux::intensive_gap::bottom::<target_name>``.

How they work
-------------

Extenders can only communicate with the architecture and the other extenders by requesting
outputs to each other.

**On model initialization**, every extender declares the outputs that it supports and their
layout, as well as the inputs that it will need.

**On model execution**, extenders receive the outputs of the model and the preceding extenders,
and they return their own outputs which the next extender can see.

By limiting the communication in this way, we ensure that extenders are modular and
their behavior is easy to predict. This helps us easily guarantee that they will
work with any architecture whose outputs are compatible with the extender's inputs, as
well as ensuring that compatible extenders can be chained together.

How to use them
---------------

The architectures that support extenders have a hyperparameter called ``extenders``
under their ``training`` section. This hyperparameter accepts a dictionary with
all the extenders that should be applied to the architecture, by order of application.

For example, the following configuration

..  code-block:: yaml

    training:
      extenders:
        extender_name_1:
          hypers_for_extender_1
        extender_name_2:
          hypers_for_extender_2
        ...

will apply ``extender_name_1`` first, then ``extender_name_2``. The hyperparameters that every
extender accepts are specific to each extender, and can be found in their documentation page.
If an extender contains the ``"outputs"`` hyperparameter, there is a shorthand for specifying
the outputs while using default values for the rest of the hyperparameters. The shorthand
is:

..  code-block:: yaml

    training:
      extenders:
        extender_name_1: whatever
        ...

which is equivalent to

..  code-block:: yaml

    training:
      extenders:
        extender_name_1:
          outputs: whatever
        ...

This improves the readability of the yaml files for typical use cases where the user
wants to use the extender to fit a target in their dataset, simply using the default
behavior of the extender.

Contribute an extender
----------------------

Extenders represent a much lower barrier entry than full architectures and the maintainance
effort required is lower. They are a great way to make published contributions accessible
to the whole ``metatrain`` community across all architectures.

Therefore, if you have a functionality that you think would be useful to the community,
don't hesitate to contribute an extender!  To get an idea of what is required, you can
check the simplest of all extenders: :ref:`the rename extender <extender-rename>`.
