import pytest

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.extenders.helpers import (
    find_all_extenders,
    get_default_hypers,
    load_extender,
    setup_extenders,
)


@pytest.fixture
def dataset_info():

    global_energy = get_energy_target_info(
        "energy",
        dict(
            sample_kind="system",
            unit="eV",
            quantity="energy",
            num_subtargets=1,
        ),
    )
    local_energy = get_energy_target_info(
        "mtt::local_energy",
        dict(
            sample_kind="atom",
            unit="eV",
            quantity="energy",
            num_subtargets=1,
        ),
    )

    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": global_energy,
            "mtt::local_energy": local_energy,
            "mtt::some_other_energy": local_energy,
        },
        extra_data={"mtt::extra_energy": local_energy},
    )


def test_find_all_extenders():
    extenders = find_all_extenders()
    assert isinstance(extenders, list)
    assert all(isinstance(extender, str) for extender in extenders)
    assert len(extenders) == 4
    assert "rename" in extenders
    assert "global_multipoles" in extenders
    assert "intensive_gap" in extenders
    assert "tensor_basis" in extenders


class TestSetupExtenders:
    def test_empty(self, dataset_info):

        hypers = {}

        dataset_info = dataset_info

        post_extenders, model_outs = setup_extenders(dataset_info, hypers)

        assert len(post_extenders) == 0
        assert model_outs == dataset_info.targets

    def test_single_extender(self, dataset_info):
        """Simplest test of an extender owning a single output, and its inputs
        are created so that the model can produce them."""

        hypers = {"rename": get_default_hypers("rename")}
        hypers["rename"]["outputs"] = "mtt::local_energy"

        dataset_info = dataset_info.copy()

        post_extenders, model_outs = setup_extenders(dataset_info, hypers)

        assert len(post_extenders) == 1
        assert "mtt::local_energy" not in model_outs

        assert "energy" in model_outs
        assert "mtt::some_other_energy" in model_outs
        assert len(model_outs) > 2

    def test_single_extender_with_inputs(self, dataset_info):
        """Test that extenders can use a target as input.

        This means that the target will flow to the loss function,
        but still we can use it as input to an extender.
        """
        hypers = {"rename": get_default_hypers("rename")}
        hypers["rename"]["outputs"] = "mtt::local_energy"
        hypers["rename"]["inputs"] = "mtt::some_other_energy"

        dataset_info = dataset_info.copy()

        post_extenders, model_outs = setup_extenders(dataset_info, hypers)

        assert len(post_extenders) == 1
        assert "mtt::local_energy" not in model_outs
        assert model_outs == {
            k: v for k, v in dataset_info.targets.items() if k != "mtt::local_energy"
        }, (
            "Extender should not add extra outputs to the model, since its inputs"
            " are already present in the model outputs."
        )

    def test_single_extender_with_inputs_from_extra_data(self, dataset_info):
        """Test that extenders can get their inputs from the extra data, and
        that in that case they don't add an output to the model."""

        hypers = {"rename": get_default_hypers("rename")}
        hypers["rename"]["outputs"] = "mtt::local_energy"
        hypers["rename"]["inputs"] = "mtt::extra_energy"

        dataset_info = dataset_info.copy()

        post_extenders, model_outs = setup_extenders(dataset_info, hypers)

        assert len(post_extenders) == 1
        assert "mtt::local_energy" not in model_outs
        assert model_outs == {
            k: v for k, v in dataset_info.targets.items() if k != "mtt::local_energy"
        }, (
            "Extender should not add extra outputs to the model, since its inputs"
            " are already present in the extra data."
        )

    def test_chained_extenders(self, dataset_info):
        """Check that an output from a previous extender can be used
        as an input by the followinf extenders."""

        hypers = {
            "rename": get_default_hypers("rename"),
            "intensive_gap": get_default_hypers("intensive_gap"),
        }

        hypers["rename"]["inputs"] = "mtt::local_energy"
        hypers["rename"]["outputs"] = "mtt::gap_bottom"
        hypers["intensive_gap"]["inputs"] = {"bottom": "mtt::gap_bottom"}
        hypers["intensive_gap"]["outputs"] = "energy"

        dataset_info = dataset_info.copy()

        post_extenders, model_outs = setup_extenders(dataset_info, hypers)

        assert len(post_extenders) == 2
        assert "energy" not in model_outs

        assert "mtt::local_energy" in model_outs
        assert "mtt::some_other_energy" in model_outs

        # There should be an extra model output for the top of the gap,
        # requested by the intensive gap extender.
        assert len(model_outs) == 3

    def test_not_available_output_error(self, dataset_info):
        """an extender is asked to use an output that doesn't exist,
        then it should raise an error.
        """

        hypers = {"intensive_gap": get_default_hypers("intensive_gap")}
        hypers["intensive_gap"]["outputs"] = "mtt::intermediate_energy"

        dataset_info = dataset_info.copy()

        with pytest.raises(ValueError, match="mtt::intermediate_energy"):
            post_extenders, model_outs = setup_extenders(dataset_info, hypers)

    def test_chained_extenders_with_laterdependency(self, dataset_info):
        """Similar to the previous test, but in this case the output will
        be known once the second extender is set up."""

        hypers = {
            "intensive_gap": get_default_hypers("intensive_gap"),
            "rename": get_default_hypers("rename"),
        }
        hypers["intensive_gap"]["outputs"] = "mtt::intermediate_energy"
        hypers["rename"]["inputs"] = "mtt::intermediate_energy"
        hypers["rename"]["outputs"] = "energy"

        dataset_info = dataset_info.copy()

        extenders, model_outs = setup_extenders(dataset_info, hypers)

        assert len(extenders) == 2
        assert isinstance(extenders[0], load_extender("intensive_gap"))
        assert isinstance(extenders[1], load_extender("rename"))
        assert "energy" not in model_outs

        assert "mtt::local_energy" in model_outs
        assert "mtt::some_other_energy" in model_outs

        # There should be two extra model outputs requested by the
        # intensive gap extender.
        assert len(model_outs) == 4
