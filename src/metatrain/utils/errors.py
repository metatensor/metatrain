class ArchitectureError(Exception):
    """
    Exception raised for errors originating from architectures

    This exception should be raised when an error occurs within an architecture's
    operation, indicating that the problem is not directly related to the
    metatrain infrastructure but rather to the specific architecture being used.

    :param exception: The original exception that was caught, which led to raising this
        custom exception.
    """

    def __init__(self, exception: Exception):
        super().__init__(
            f"{exception.__class__.__name__}: {exception}\n\n"
            "The error above most likely originates from an architecture.\n\n"
            "If you think this is a bug, please contact its maintainer (see the "
            "architecture's documentation) and include the full traceback error.log."
        )


class OutOfMemoryError(Exception):
    """
    Exception raised for out of memory errors.

    This exception should be raised when a GPU out of memory error occurs,
    indicating that the model or batch size is too large for the available GPU memory.

    :param exception: The original exception that was caught, which led to raising this
        custom exception.
    """

    def __init__(self, exception: Exception):
        super().__init__(
            f"{exception}\n\n"
            "The error above likely means that the model ran out of memory during "
            "training. You can try to reduce the batch size or reduce the model size "
            "(e.g., reduce the number of features or layers). If available check the "
            "architecture's documentation for more suggestions."
        )
