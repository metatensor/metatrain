.. _advanced_base_conf:

Advanced Base Configuration
===========================

Here, we show how some advanced base properties in the ``options.yaml`` can
be adjusted. They should be written without indentation in the ``options.yaml`` file.

:param device: The device in which the training should be run. Takes two possible
    values: ``cpu`` and ``gpu``. Default: ``cpu``
:param base_precision: Override the base precision of all floats during training. By
    default an optimal precision is obtained from the architecture. Changing this will
    have an effect on the memory consumption during training and maybe also on the
    accuracy of the model. Possible values: ``64``, ``32`` or ``16``.
:param seed: Seed used to start the training. Set all the seeds
    of ``numpy.random``, ``random``, ``torch`` and ``torch.cuda`` (if available)
    to the same value ``seed``.
    If ``seed=None`` all the seeds are set to a random number. Default: ``None``
    Note: in a ``.yaml`` file ``None`` is ``null``.

In the next tutorials we show how to override the default parameters of an architecture.
