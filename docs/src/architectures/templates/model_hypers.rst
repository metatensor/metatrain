.. _arch-{{architecture}}_model_hypers:

Model hyperparameters
------------------------

{% if model_hypers %}
The parameters that go under the ``architecture.model`` section of the config file
are the following:

.. container:: mtt-hypers-remove-classname

    ..

    {% for hyper in model_hypers %}
        .. autoattribute:: {{model_hypers_path}}.{{hyper}}

    {% endfor %}
{% else %}
This architecture has no model hyperparameters: there is nothing to set under
the ``architecture.model`` section of the config file.
{% endif %}
