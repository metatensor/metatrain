import torch


def get_radial_mask(r, r_cut: float, r_transition: float):
    # All radii are already guaranteed to be smaller than r_cut
    return torch.where(
        r < r_transition,
        torch.ones_like(r),
        0.5 * (torch.cos(torch.pi * (r - r_transition) / (r_cut - r_transition)) + 1.0),
    )
