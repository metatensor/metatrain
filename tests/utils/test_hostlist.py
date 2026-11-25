import pytest

from metatrain.utils.distributed.hostlist import expand_hostlist


@pytest.mark.parametrize(
    "nodelist, expected",
    [
        ("node1", ["node1"]),
        ("node1,node2", ["node1", "node2"]),
        ("nid[001-003]", ["nid001", "nid002", "nid003"]),
        ("node[8-11]", ["node8", "node9", "node10", "node11"]),
        ("node[01-02,7]", ["node01", "node02", "node7"]),
        ("a[1-2],b3,c[09-10]", ["a1", "a2", "b3", "c09", "c10"]),
        ("r[1-2]n[1-2]", ["r1n1", "r1n2", "r2n1", "r2n2"]),
        ("gpu-[1,3]-ib", ["gpu-1-ib", "gpu-3-ib"]),
    ],
)
def test_expand_hostlist(nodelist, expected):
    assert expand_hostlist(nodelist) == expected


@pytest.mark.parametrize("nodelist", ["node[3-1]", "node[1-2", "node1-2]"])
def test_expand_hostlist_invalid(nodelist):
    with pytest.raises(ValueError):
        expand_hostlist(nodelist)
