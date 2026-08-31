.. _arch-{{architecture}}_trainer_hypers:

Trainer hyperparameters
-------------------------

{% if trainer_hypers %}
The parameters that go under the ``architecture.trainer`` section of the config file
are the following:

.. container:: mtt-hypers-remove-classname

    ..

    {% for hyper in trainer_hypers %}
        .. autoattribute:: {{trainer_hypers_path}}.{{hyper}}

    {% endfor %}
{% else %}
This architecture has no trainer hyperparameters: there is nothing to set under
the ``architecture.trainer`` section of the config file.
{% endif %}
