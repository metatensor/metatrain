from pathlib import Path

from metatensor.torch.atomistic import NeighborListOptions

from metatrain.utils.data.readers.systems import read_systems_ase
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


RESOURCES_PATH = Path(__file__).parents[1] / "resources"


def test_attach_neighbor_lists():
    filename = RESOURCES_PATH / "qm9_reduced_100.xyz"
    systems = read_systems_ase(filename)

    requested_neighbor_lists = [
        NeighborListOptions(cutoff=4.0, full_list=True),
        NeighborListOptions(cutoff=5.0, full_list=False),
        NeighborListOptions(cutoff=6.0, full_list=True),
    ]

    new_system = get_system_with_neighbor_lists(systems[0], requested_neighbor_lists)

    assert requested_neighbor_lists[0] in new_system.known_neighbor_lists()
    assert requested_neighbor_lists[1] in new_system.known_neighbor_lists()
    assert requested_neighbor_lists[2] in new_system.known_neighbor_lists()

    extraneous_nl = NeighborListOptions(cutoff=5.0, full_list=True)
    assert extraneous_nl not in new_system.known_neighbor_lists()

    for nl_options in new_system.known_neighbor_lists():
        nl = new_system.get_neighbor_list(nl_options)
        assert nl.samples.names == [
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ]
        assert len(nl.values.shape) == 3
