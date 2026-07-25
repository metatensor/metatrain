import pytest

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.hooks.helpers import find_all_hooks, setup_post_hooks


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


def test_find_all_hooks():
    hooks = find_all_hooks()
    assert isinstance(hooks, list)
    assert all(isinstance(hook, str) for hook in hooks)
    assert len(hooks) == 4
    assert "identity" in hooks
    assert "global_multipoles" in hooks
    assert "intensive_gap" in hooks
    assert "tensor_basis" in hooks


class TestSetupPostHooks:
    def test_empty(self, dataset_info):

        hypers = {}

        post_hooks, model_outs = setup_post_hooks(hypers, dataset_info)

        assert len(post_hooks) == 0
        assert model_outs == dataset_info.targets

    def test_single_hook(self, dataset_info):
        """Simplest test of a hook owning a single output, and its inputs
        are created so that the model can produce them."""

        hypers = {
            "identity": "mtt::local_energy",
        }

        post_hooks, model_outs = setup_post_hooks(hypers, dataset_info)

        assert len(post_hooks) == 1
        assert "mtt::local_energy" not in model_outs

        assert "energy" in model_outs
        assert "mtt::some_other_energy" in model_outs
        assert len(model_outs) > 2

    def test_single_hook_with_inputs(self, dataset_info):
        """Test that hooks can use a target as input.

        This means that the target will flow to the loss function,
        but still we can use it as input to a hook.
        """
        hypers = {
            "identity": {
                "outputs": "mtt::local_energy",
                "inputs": "mtt::some_other_energy",
            },
        }

        post_hooks, model_outs = setup_post_hooks(hypers, dataset_info)

        assert len(post_hooks) == 1
        assert "mtt::local_energy" not in model_outs
        assert model_outs == {
            k: v for k, v in dataset_info.targets.items() if k != "mtt::local_energy"
        }, (
            "Hook should not add extra outputs to the model, since its inputs"
            " are already present in the model outputs."
        )

    def test_single_hook_with_inputs_from_extra_data(self, dataset_info):
        """Test that hooks can get their inputs from the extra data, and
        that in that case they don't add an output to the model."""

        hypers = {
            "identity": {"outputs": "mtt::local_energy", "inputs": "mtt::extra_energy"},
        }

        post_hooks, model_outs = setup_post_hooks(hypers, dataset_info)

        assert len(post_hooks) == 1
        assert "mtt::local_energy" not in model_outs
        assert model_outs == {
            k: v for k, v in dataset_info.targets.items() if k != "mtt::local_energy"
        }, (
            "Hook should not add extra outputs to the model, since its inputs"
            " are already present in the extra data."
        )

    def test_chained_hooks(self, dataset_info):
        """Check that an output from a previous hook can be used
        as an input by the followinf hooks."""

        hypers = {
            "identity": {"outputs": "mtt::gap_bottom", "inputs": "mtt::local_energy"},
            "intensive_gap": {
                "outputs": "energy",
                "inputs": {
                    "bottom": "mtt::gap_bottom",
                },
            },
        }

        post_hooks, model_outs = setup_post_hooks(hypers, dataset_info)

        assert len(post_hooks) == 2
        assert "energy" not in model_outs

        assert "mtt::local_energy" in model_outs
        assert "mtt::some_other_energy" in model_outs

        # There should be an extra model output for the top of the gap,
        # requested by the intensive gap hook.
        assert len(model_outs) == 3

    def test_not_available_output_error(self, dataset_info):
        """A hook is asked to use an output that doesn't exist,
        then it should raise an error.
        """

        hypers = {
            "intensive_gap": {
                "outputs": "mtt::intermediate_energy",
            },
        }

        with pytest.raises(ValueError, match="mtt::intermediate_energy"):
            post_hooks, model_outs = setup_post_hooks(hypers, dataset_info)

    def test_chained_hooks_with_laterdependency(self, dataset_info):
        """Similar to the previous test, but in this case the output will
        be known once the second hook is set up."""

        hypers = {
            "intensive_gap": {
                "outputs": "mtt::intermediate_energy",
            },
            "identity": {"outputs": "energy", "inputs": "mtt::intermediate_energy"},
        }

        post_hooks, model_outs = setup_post_hooks(hypers, dataset_info)

        assert len(post_hooks) == 2
        assert "energy" not in model_outs

        assert "mtt::local_energy" in model_outs
        assert "mtt::some_other_energy" in model_outs

        # There should be two extra model outputs requested by the
        # intensive gap hook.
        assert len(model_outs) == 4
