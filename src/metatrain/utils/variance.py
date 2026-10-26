"""Helpers for heteroscedastic (aleatoric) variance outputs.

A model can predict, next to a target ``target``, the variance of the noise on that
target. This variance is exposed as the auxiliary output
``mtt::aux::{target}_variance``, which has the same metadata as the target itself.
"""


def get_variance_output_name(target_name: str) -> str:
    """
    Get the name of the auxiliary output holding the predicted variance of a target.

    :param target_name: Name of the target.
    :return: Name of the corresponding variance output.
    """
    base_name = target_name.replace("mtt::", "")
    return f"mtt::aux::{base_name}_variance"
