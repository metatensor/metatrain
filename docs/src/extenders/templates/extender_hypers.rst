.. _extender-{{extender}}_hypers:

Extender hyperparameters
------------------------

{% if extender_hypers %}
The default hyperparameters for this extender are:

.. literalinclude:: {{default_hypers_path}}
   :language: yaml

and here is the documentation for each hyperparameter:

.. container:: mtt-hypers-remove-classname

    ..

    {% for hyper in extender_hypers %}
        .. autoattribute:: {{extender_hypers_path}}.{{hyper}}

    {% endfor %}
{% else %}
This extender has no hyperparameters.
{% endif %}
