from typing import Dict


def update_state_dict(state_dict: Dict) -> Dict:
    """
    Updates the state_dict keys so they match the model's keys.
    """
    new_state_dict = {}
    for name, value in state_dict.items():
        if "pet_model." in name:
            name = name.split("pet_model.")[1]
        new_state_dict[name] = value
    return new_state_dict
