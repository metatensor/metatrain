from pathlib import Path

from metatensor.torch.atomistic import NeighborsListOptions

from metatensor.models.utils.data.readers.systems import read_systems_ase
from metatensor.models.utils.neighbors_lists import get_system_with_neighbors_lists


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


def test_attach_neighbor_lists():
    filename = RESOURCES_PATH / "qm9_reduced_100.xyz"
    systems = read_systems_ase(filename)

    requested_neighbor_lists = [
        NeighborsListOptions(model_cutoff=4.0, full_list=True),
        NeighborsListOptions(model_cutoff=5.0, full_list=False),
        NeighborsListOptions(model_cutoff=6.0, full_list=True),
    ]

    new_system = get_system_with_neighbors_lists(systems[0], requested_neighbor_lists)

    assert requested_neighbor_lists[0] in new_system.known_neighbors_lists()
    assert requested_neighbor_lists[1] in new_system.known_neighbors_lists()
    assert requested_neighbor_lists[2] in new_system.known_neighbors_lists()

    extraneous_nl = NeighborsListOptions(model_cutoff=5.0, full_list=True)
    assert extraneous_nl not in new_system.known_neighbors_lists()

    for nl_options in new_system.known_neighbors_lists():
        nl = new_system.get_neighbors_list(nl_options)
        assert nl.samples.names == [
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ]
        assert len(nl.values.shape) == 3
