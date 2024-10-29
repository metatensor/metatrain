def update_state_dict(state_dict: dict) -> dict:
    """
    Updates the state_dict keys so they match the model's keys.
    """
    new_state_dict = {}
    for name, value in state_dict.items():
        name = name.split("pet_model.")[1]
        new_state_dict[name] = value
    return state_dict
