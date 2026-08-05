.. _hook-{{hook}}_hypers:

Hook hyperparameters
------------------------

{% if hook_hypers %}
The default hyperparameters for this hook are:

.. literalinclude:: {{default_hypers_path}}
   :language: yaml

and here is the documentation for each hyperparameter:

.. container:: mtt-hypers-remove-classname

    ..

    {% for hyper in hook_hypers %}
        .. autoattribute:: {{hook_hypers_path}}.{{hyper}}

    {% endfor %}
{% else %}
This hook has no hyperparameters.
{% endif %}
