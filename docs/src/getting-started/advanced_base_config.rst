.. _advanced_base_conf:

Advanced Base Configuration
===========================

Here, we explain how some advanced base properties in the ``options.yaml`` can
be adjusted. They should be written without indentation in the ``options.yaml`` file.

:param device: The device in which the training should be run. Takes two possible
    values: ``cpu`` and ``gpu``. Default: ``cpu``
:param seed: Seed used to start the training. Set all the seeds
    of ``numpy.random``, ``random``, ``torch`` and ``torch.cuda`` (if available)
    to the same value ``seed``.
    If ``seed=-1`` all the seeds are set to a random number. Default: ``-1``
:param base_precision: This may increase the accuracy improvements but will increase the
    memory consumption during training. Default: ``64``
