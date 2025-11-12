"""
Tools to generate hyperparameter specifications for MACE models.
"""

from pathlib import Path

from metatrain.experimental.mace.utils.hypers import (
    MACE_MODEL_ARG_KEYS,
    get_mace_hypers_spec
)

def write_mace_hypers_spec():
    """Writes the MACE hyperparameter specification to a YAML file.

    It extracts the hyperparameter specification from the MACE argparser
    and writes it to mace_hypers_spec.yaml.
    """
    
    with open(Path(__file__).parent / "documentation_template.py", "r") as f:
        template_content = f.read()

    def _get_default(spec):
        if "list" in spec["type"]:
            return spec["default"]
        elif spec["default"] == "None":
            return "None"
        return repr(spec["default"])

    hypers_spec = get_mace_hypers_spec()
    mace_hypers_str = ""
    for key in MACE_MODEL_ARG_KEYS:
        spec = hypers_spec[key]
        mace_hypers_str += f'    {key}: {spec["type"]} = {_get_default(spec)}\n    """{spec["help"]}""" \n'

    template_content = template_content.format(
        mace_hypers=mace_hypers_str,
        **{f'mace_param_{key}': f'{hypers_spec[key]["type"]} = {_get_default(hypers_spec[key])}' for key in hypers_spec},
        **{f'mace_help_{key}': hypers_spec[key]["help"] for key in hypers_spec}
    )

    with open(Path(__file__).parent.parent / "documentation.py", "w") as f:
        f.write(template_content)

if __name__ == "__main__":
    write_mace_hypers_spec()