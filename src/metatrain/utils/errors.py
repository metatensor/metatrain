class ArchitectureError(Exception):
    """
    Exception raised for errors originating from architectures

    This exception should be raised when an error occurs within an architecture's
    operation, indicating that the problem is not directly related to the
    metatrain infrastructure but rather to the specific architecture being used.

    :param exception: The original exception that was caught, which led to raising this
        custom exception.
    """

    def __init__(self, exception):
        super().__init__(
            "The error below most likely originates from an architecture. If you think "
            "this is a bug, please contact its maintainer (see the architecture's "
            f"documentation).\n\n{exception}"
        )
