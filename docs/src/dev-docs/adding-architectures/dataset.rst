Using Dataset
#############

The ``metatensor-models`` package provides a set of parser for various
storage formats for atomic data. These parsers will return the data as
:py:class:`metatensor.models.utils.data.Dataset` objects, which is a simple
class that inherits from :py:class:`torch.utils.data.Dataset`. 

This dataset is an indexable collection of data. When indexed, it will
return a ``Tuple[System, Dict[str, TensorMap]]``, which is effectively
a tuple containing the inputs and the outputs of the model in the
``metatensor`` format.

You can either extract data yourself from the ``Dataset`` class and
convert it to a format that is more suited to your model (see the 
``System`` and ``TensorMap`` sections of this guide), or you can use
the provided :py:function:`metatensor.models.utils.data.collate_fn` to build
:py:class:`torch.utils.data.DataLoader` objects that will return the data in
the ``Tuple[List[System], Dict[str, TensorMap]]`` format, which
corresponds exactly to the input and output of a ``metatensor`` model.
