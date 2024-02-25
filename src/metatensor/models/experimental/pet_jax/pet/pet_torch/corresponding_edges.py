import torch


def get_corresponding_edges(array):
    n_edges = len(array)
    array_inversed = array.flip(1)
    inverse_indices = torch.empty((n_edges,), dtype=torch.long)
    for i in range(n_edges):
        inverse_indices[i] = torch.nonzero(
            torch.all(array_inversed == array[i], dim=1), as_tuple=False
        )[0]
    return inverse_indices
