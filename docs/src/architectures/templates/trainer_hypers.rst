.. _architecture-{{architecture}}_trainer_hypers:

Trainer hyperparameters
-------------------------

The parameters that go under the ``architecture.trainer`` section of the config file
are the following:

.. container:: mtt-hypers-remove-classname

    ..

    {% for hyper in trainer_hypers %}
        .. autoattribute:: {{trainer_hypers_path}}.{{hyper}}

    {% endfor %}
