.. _architecture-{{architecture}}_model_hypers:

Model hyperparameters
------------------------

The parameters that go under the ``architecture.model`` section of the config file
are the following:

.. container:: mtt-hypers-remove-classname

    ..

    {% for hyper in model_hypers %}
        .. autoattribute:: {{model_hypers_path}}.{{hyper}}

    {% endfor %}
