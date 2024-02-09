Using ModelCapabilities
#######################

If you are not looking to integrate advanced features like fine-tuning a model
on different datasets than the one it was trained on, you simply ignore the
``ModelCapabilities`` class and simply feed it to the model. This class will help
atomistic simulation engines check if the model is compatible with the data
they are trying to simulate.

For more information, see :py:class:`metatensor.torch.atomistic.ModelCapabilities`.
