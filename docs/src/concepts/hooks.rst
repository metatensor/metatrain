.. _concept-hooks:

Using hooks
===========

Hooks are a powerful way to extend the functionality of architectures in ``metatrain``.

For a list of available hooks and their functionality, see :ref:`available-hooks`.

.. warning::

    Hooks are a big infrastructural change that has been recently added
    to ``metatrain``. As such, it is possible that it contains bugs or that
    the usage changes in the future. If you find any issues or have suggestions,
    feedback is highly appreciated.

What hooks do
-------------

They implement functions that are called in the forward pass, after the core of the
model. Their purpose is to augment the capabilities of all architectures in a modular
way, so that the functionality is immediately available to all architectures without
having to modify their code. Any new architecture that is added to ``metatrain``
will automatically benefit from the hooks.

Tipical examples of hooks are:

- Readout layers that implement different pooling strategies (e.g.
  :ref:`the intensive gap hook <hook-intensive_gap>` implements a pooling strategy that is
  invariant to the number of atoms in the system).
- Modules that produce outputs that the architecture does not natively support
  (e.g. :ref:`the tensor basis hook <hook-tensor_basis>` produces spherical outputs
  from scalar outputs of the architecture).


Usually, the way hooks work is that you indicate the target of your dataset that
the hook should produce, and then the hook will automatically request the necessary
intermediate outputs from the architecture. However, one can do more complex stuff
like chaining hooks or specifying the outputs of the architecture that the hook
should use as input. The exact functionality really depends on each particular hook.

How they work
-------------

Hooks can only communicate with the architecture and the other hooks by requesting
outputs to each other.

**On initialization**, every hook declares the outputs that it supports and their
layout, as well as the inputs that it will need.

**On forward**, hooks receive the outputs of the model and the preceding hooks,
and they return their own outputs which the next hook can see.

By limiting the communication in this way, we ensure that hooks are modular and
their behavior is easy to predict. This helps us easily guarantee that they will
work with any architecture whose outputs are compatible with the hook's inputs, as
well as ensuring that compatible hooks can be chained together.

How to use them
---------------

The architectures that support hooks have a hyperparameter called ``forward_hooks``
under their ``training`` section. This hyperparameter accepts a dictionary with
all the hooks that should be applied to the architecture, by order of application.

For example, the following configuration

..  code-block:: yaml

    training:
      forward_hooks:
        hook_name_1:
          hypers_for_hook_1
        hook_name_2:
          hypers_for_hook_2
        ...

will apply ``hook_name_1`` first, then ``hook_name_2``. The hyperparameters that every
hook accepts are specific to each hook, and can be found in their documentation page.
If a hook contains the ``"outputs"`` hyperparameter, there is a shorthand for specifying
the outputs while using default values for the rest of the hyperparameters. The shorthand
is:

..  code-block:: yaml

    training:
      forward_hooks:
        hook_name_1: whatever
        ...

which is equivalent to

..  code-block:: yaml

    training:
      forward_hooks:
        hook_name_1:
          outputs: whatever
        ...

This improves the readability of the yaml files for typical use cases where the user
wants to use the hook to fit a target in their dataset, simply using the default
behavior of the hook.
