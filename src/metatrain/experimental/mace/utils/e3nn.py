import inspect
from functools import wraps

from e3nn.nn import Activation


def patch_e3nn():
    """Patch e3nn to make it torchscript compatible.

    The old versions of e3nn were missing a type annotation
    and this made models not torchscriptable.
    """

    # We need to add an int type annotation for the 'dim' argument of Activation.forward
    # (this is a problem for example in e3nn==0.4.4, which is the one used
    # in MACE for now: https://github.com/ACEsuit/mace/issues/555)
    # The fix is quite involved because torch.jit.script directly reads the source code.
    # Luckily, torch uses the inspect module to get it
    # (https://github.com/pytorch/pytorch/blob/main/torch/_sources.py#L24) so we can
    # wrap the inspect.getsourcelines function to modify its behavior.

    # Get the real getsourcelines function from the inspect module
    _real_getsourcelines = inspect.getsourcelines

    # Function to check if the object is Activation.forward
    def _is_activation_forward(object):
        """Check if the object is Activation.forward method,
        either unbound or bound."""
        return (
            hasattr(object, "__func__") and object.__func__ is Activation.forward
        ) or object is Activation.forward

    # Wrapper around inspect.getsourcelines that modifies the returned
    # source code if the object is Activation.forward.
    @wraps(_real_getsourcelines)
    def wrapped_getsourcelines(object):
        # Call original function
        lines, lineno = _real_getsourcelines(object)

        # If the object is Activation.forward, modify the relevant line
        if _is_activation_forward(object):
            for i, line in enumerate(lines):
                if "def forward(self, features, dim=-1):" in line:
                    lines[i] = line.replace(
                        "def forward(self, features, dim=-1):",
                        "def forward(self, features, dim: int = -1):",
                    )
                    break

        # Return (possibly modified) source code
        return lines, lineno

    inspect.getsourcelines = wrapped_getsourcelines
