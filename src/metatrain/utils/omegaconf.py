import json
from typing import Any, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.basecontainer import BaseContainer

from .. import PACKAGE_ROOT, RANDOM_SEED
from .architectures import import_architecture
from .devices import pick_devices
from .jsonschema import validate


def _get_architecture_model(conf: BaseContainer) -> Any:
    architecture = import_architecture(conf["architecture"]["name"])
    return architecture.__model__


def default_device(_root_: BaseContainer) -> str:
    """
    Custom OmegaConf resolver to find the default device of an architecture.

    Device is found using the :py:func:metatrain.utils.devices.pick_devices`
    function.

    :param _root_: The root configuration.
    :return: The default device as a string. If multiple devices are found, returns
        "multi-cuda".
    """

    Model = _get_architecture_model(_root_)
    desired_device = pick_devices(Model.__supported_devices__)

    if len(desired_device) > 1:
        return "multi-cuda"
    else:
        return desired_device[0].type


def default_precision(_root_: BaseContainer) -> int:
    """
    Custom OmegaConf resolver to find the default precision of an architecture.

    File format is obtained based on the architecture name and its first entry in the
    ``supported_dtypes`` list.

    :param _root_: The root configuration.
    :return: The default precision in bits (16, 32, or 64).
    """

    Model = _get_architecture_model(_root_)

    # desired `dtype` is the first entry
    default_dtype = Model.__supported_dtypes__[0]

    # `base_precision` in options has to be a integer and not a torch.dtype
    if default_dtype in [torch.float64, torch.double]:
        return 64
    elif default_dtype == torch.float32:
        return 32
    elif default_dtype == torch.float16:
        return 16
    else:
        raise ValueError(
            f"architectures `default_dtype` ({default_dtype}) refers to an unknown "
            "torch dtype. This should not happen."
        )


def default_huber_loss_delta() -> float:
    """
    Return the default delta for the huber loss.

    :return: The default delta for the huber loss.
    """
    return 1.0


# Register custom resolvers
OmegaConf.register_new_resolver("default_device", default_device)
OmegaConf.register_new_resolver("default_precision", default_precision)
OmegaConf.register_new_resolver("default_random_seed", lambda: RANDOM_SEED)
OmegaConf.register_new_resolver("default_loss_type", lambda: "mse")
OmegaConf.register_new_resolver("default_loss_reduction", lambda: "mean")
OmegaConf.register_new_resolver("default_loss_weight", lambda: 1.0)


def _resolve_single_str(config: str) -> DictConfig:
    return OmegaConf.create({"read_from": config})


# Base options/configurations
BASE_OPTIONS = OmegaConf.create(
    {
        "device": "${default_device:}",
        "base_precision": "${default_precision:}",
        "seed": "${default_random_seed:}",
    }
)


CONF_SYSTEMS = OmegaConf.create(
    {
        "read_from": "${..read_from}",
        "reader": None,
        "length_unit": None,
    }
)

CONF_TARGET_FIELDS = OmegaConf.create(
    {
        "quantity": "energy",
        "read_from": "${...systems.read_from}",
        "reader": None,
        "key": None,
        "unit": None,
        "per_atom": False,
        "type": "scalar",
        "num_subtargets": 1,
    }
)

CONF_EXTRA_FIELDS = OmegaConf.create(
    {
        "quantity": "",
        "read_from": "${...systems.read_from}",
        "reader": None,
        "key": None,
        "unit": None,
        "per_atom": False,
        "type": "scalar",
        "num_subtargets": 1,
    }
)

CONF_GRADIENTS = OmegaConf.create({"forces": False, "stress": False, "virial": False})
CONF_GRADIENT = OmegaConf.create(
    {
        "read_from": "${..read_from}",
        "reader": None,
        "key": None,
    }
)

CONF_LOSS = OmegaConf.create(
    {
        "type": "${default_loss_type:}",
        "weight": "${default_loss_weight:}",
        "reduction": "${default_loss_reduction:}",
        "gradients": {},
    }
)

KNOWN_GRADIENTS = list(CONF_GRADIENTS.keys())

# Merge configs to get default configs for energies and other targets
CONF_TARGET = OmegaConf.merge(CONF_TARGET_FIELDS, CONF_GRADIENTS)
CONF_ENERGY = CONF_TARGET.copy()
CONF_ENERGY["forces"] = CONF_GRADIENT.copy()
CONF_ENERGY["stress"] = CONF_GRADIENT.copy()
CONF_EXTRA_DATA = CONF_EXTRA_FIELDS.copy()

# Schema with the dataset options
with open(PACKAGE_ROOT / "share/schema-dataset.json") as f:
    SCHEMA_DATASET = json.load(f)


def check_dataset_options(dataset_config: ListConfig) -> None:
    """Perform consistency checks within one dataset config.

    This is useful if the dataset config is made of several datasets.

    - The function checks if ``length_units`` in each system section are known and the
       same.
    - For unknown quantities a warning is given.
    - If the names of the ``"targets"`` sections are the same between the elements of
       the list of datasets also the units must be the same.
    - Two targets with the names `{target}` and `mtt::{target}` are not allowed.

    :param dataset_config: A List of configuration to be checked. In the list contains
        only one element no checks are performed.
    :raises ValueError: If the units are not consistent between the dataset options or
        if two different targets have the `{target}` and `mtt::{target}` names.
    """
    desired_config = dataset_config[0]

    if hasattr(desired_config, "targets"):
        # save unit for each target seaction for later comparison
        unit_dict = {k: v["unit"] for k, v in desired_config["targets"].items()}
    else:
        unit_dict = {}

    if hasattr(desired_config, "systems"):
        desired_length_unit = desired_config["systems"]["length_unit"]
    else:
        desired_length_unit = None

    if hasattr(desired_config, "extra_data"):
        # save unit for each extra_data section for later comparison
        for extra_data_key, extra_data in desired_config["extra_data"].items():
            unit_dict[extra_data_key] = extra_data["unit"]

    # loop over ALL configs because we have check units for all elements in
    # `dataset_config`
    for actual_config in dataset_config:
        if desired_length_unit:
            # Perform consistency checks between config elements
            actual_length_unit = actual_config["systems"]["length_unit"]
            if actual_length_unit != desired_length_unit:
                raise ValueError(
                    "`length_unit`s are inconsistent between one of the dataset "
                    f"options. {actual_length_unit!r} != {desired_length_unit!r}."
                )

        if hasattr(actual_config, "targets"):
            for target_key, target in actual_config["targets"].items():
                unit = target["unit"]

                # If a target section name is not part of the saved units we add it for
                # later comparison. We do not have to start the loop again because this
                # target section name is not present in one of the datasets checked
                # before.
                if target_key not in unit_dict.keys():
                    unit_dict[target_key] = unit

                if unit_dict[target_key] != unit:
                    raise ValueError(
                        f"Units of target section {target_key!r} are inconsistent. "
                        f"Found {unit!r} and {unit_dict[target_key]!r}!"
                    )

        # `target` and `mtt::target` are not allowed to be present at the same time
        if hasattr(actual_config, "targets"):
            for target_key in actual_config["targets"].keys():
                if f"mtt::{target_key}" in actual_config["targets"].keys():
                    raise ValueError(
                        f"Two targets with the names `{target_key}` and "
                        f"`mtt::{target_key}` are not allowed to be present "
                        "at the same time."
                    )

        if hasattr(actual_config, "extra_data"):
            for extra_data_key, extra_data in actual_config["extra_data"].items():
                unit = extra_data["unit"]

                # If a extra_data section name is not part of the saved units we add it
                # for later comparison. We do not have to start the loop again because
                # this extra_data section name is not present in one of the datasets
                # checked before.
                if extra_data_key not in unit_dict.keys():
                    unit_dict[extra_data_key] = unit

                if unit_dict[extra_data_key] != unit:
                    raise ValueError(
                        f"Units of extra_data section {extra_data_key!r} are "
                        "inconsistent. "
                        f"Found {unit!r} and {unit_dict[extra_data_key]!r}!"
                    )


def expand_dataset_config(conf: Union[str, DictConfig, ListConfig]) -> ListConfig:
    """Expands shorthand notations in a dataset configuration to its full format.

    This function takes a dataset configuration, either as a :py:class:str,
    :py:class:`omegaconf.DictConfig` or a :py:class:`omegaconf.ListConfig`, and expands
    it into a detailed configuration format. It processes systems, targets, and gradient
    sections, setting default values and inferring missing information. Unknown keys are
    ignored, allowing for flexibility.

    If the dataset configuration is either a :class:`str` or a
    :class:`omegaconf.DictConfig`

    The function performs the following steps for each config

    - Merges and interpolates the input configuration with the base configurations.
    - Expands shorthand notations like file paths or simple true/false settings to full
      dictionary systems. This includes setting the units to the base units of
      ``"angstrom"`` and ``"eV"``.
    - Handles special cases, such as the mandatory nature of the "energy" section for MD
      simulations and the mutual exclusivity of 'stress' and 'virial' sections.
      Additionally the gradient sections for "forces" are enables by default.

    :param conf: The dataset configuration, either as a file path string or a DictConfig
        object.
    :raises ValueError: If both ``virial`` and ``stress`` sections are enabled in the
        "energy" target, as this is not permissible for training.
    :return: List of datasets configurations. If ``conf`` was a :class:`str` or a
        :class:`omegaconf.DictConfig` the list contains only a single element.
    """
    # Expand str -> DictConfig
    if isinstance(conf, str):
        read_from = conf
        conf = OmegaConf.create(
            {"systems": read_from, "targets": {"energy": read_from}}
        )

    # Expand DictConfig -> ListConfig
    if isinstance(conf, DictConfig):
        conf = OmegaConf.create([conf])

    # Perform expansion per config inside the ListConfig
    for conf_element in conf:
        validate(instance=OmegaConf.to_container(conf_element), schema=SCHEMA_DATASET)
        if hasattr(conf_element, "systems"):
            if type(conf_element["systems"]) is str:
                conf_element["systems"] = _resolve_single_str(conf_element["systems"])

            conf_element["systems"] = OmegaConf.merge(
                CONF_SYSTEMS, conf_element["systems"]
            )

        if hasattr(conf_element, "targets"):
            for target_key, target in conf_element["targets"].items():
                if type(target) is str:
                    target = _resolve_single_str(target)

                # for special case "energy" we enable sections for `forces` and `stress`
                # gradients by default
                if target_key == "energy":
                    target = OmegaConf.merge(CONF_ENERGY, target)
                else:
                    target = OmegaConf.merge(CONF_TARGET, target)

                if target["key"] is None:
                    target["key"] = target_key

                # update DictConfig to allow for config node interpolation
                conf_element["targets"][target_key] = target

                # merge and interpolate possibly present gradients with default gradient
                # config
                for gradient_key, gradient_conf in conf_element["targets"][
                    target_key
                ].items():
                    if gradient_key in KNOWN_GRADIENTS:
                        if gradient_conf is True:
                            gradient_conf = CONF_GRADIENT.copy()
                        elif type(gradient_conf) is str:
                            gradient_conf = _resolve_single_str(gradient_conf)

                        if isinstance(gradient_conf, DictConfig):
                            gradient_conf = OmegaConf.merge(
                                CONF_GRADIENT, gradient_conf
                            )

                            if gradient_conf["key"] is None:
                                gradient_conf["key"] = gradient_key

                            conf_element["targets"][target_key][gradient_key] = (
                                gradient_conf
                            )

                # If user sets the virial gradient and leaves the stress gradient
                # untouched, we disable the by default enabled stress gradient section.
                base_stress_gradient_conf = CONF_GRADIENT.copy()
                base_stress_gradient_conf["key"] = "stress"

                if (
                    target_key == "energy"
                    and conf_element["targets"][target_key]["virial"]
                    and conf_element["targets"][target_key]["stress"]
                    == base_stress_gradient_conf
                ):
                    conf_element["targets"][target_key]["stress"] = False

                if (
                    conf_element["targets"][target_key]["stress"]
                    and conf_element["targets"][target_key]["virial"]
                ):
                    raise ValueError(
                        f"Cannot perform training with respect to virial and stress "
                        f"as in section {target_key}. Set either `virial: off` or "
                        "`stress: off`."
                    )

        if hasattr(conf_element, "extra_data"):
            for extra_data_key, extra_data in conf_element["extra_data"].items():
                if type(extra_data) is str:
                    extra_data = _resolve_single_str(extra_data)

                extra_data = OmegaConf.merge(CONF_EXTRA_DATA, extra_data)

                if extra_data["key"] is None:
                    extra_data["key"] = extra_data_key

                # update DictConfig to allow for config node interpolation
                conf_element["extra_data"][extra_data_key] = extra_data

    check_dataset_options(conf)
    return conf


def expand_loss_config(conf: DictConfig) -> DictConfig:
    """Expand the loss configuration to fully specify loss terms for different targets
    and their gradients.

    This function applies this precedence order when expanding the loss configuration:

      Per-target user loss  >  top-level shorthands  >  defaults (as in CONF_LOSS)

    For energy targets, these shorthands are allowed at the top level of the loss
    configuration, and expanded into the per-target loss as follows:

      - forces         -> gradients.positions
      - stress/virial  -> gradients.strain

    For non-energy targets, there are no top-level shorthands, and the loss must be
    fully specified in the per-target loss section.

    :param conf: The loss configuration to expand.
    :return: The expanded loss configuration.
    """

    def _finalize_huber(node: DictConfig) -> None:
        """
        Ensure Huber losses have a delta on the target node and its immediate gradients.
        :param node: A DictConfig node representing a loss configuration.
        """

        def _apply(n: DictConfig) -> None:
            if (
                isinstance(n, (dict, DictConfig))
                and n.get("type") == "huber"
                and "delta" not in n
            ):
                try:
                    n["delta"] = default_huber_loss_delta()
                except Exception:
                    n["delta"] = 1.0

        _apply(node)
        g = node.get("gradients")
        if isinstance(g, (dict, DictConfig)):
            for gv in g.values():
                _apply(gv)

    def _inherit_parent_into_gradients(parent: DictConfig) -> None:
        """
        Copy missing (type, weight, reduction, delta-if-present) from parent into
        existing gradient sections.

        :param parent: A DictConfig node representing a loss configuration.
        """
        if not isinstance(parent, (dict, DictConfig)):
            return
        g = parent.get("gradients")
        if not isinstance(g, (dict, DictConfig)):
            return

        keys = ["type", "weight", "reduction"]
        if "delta" in parent:
            keys.append("delta")

        for gv in g.values():
            if isinstance(gv, (dict, DictConfig)):
                for k in keys:
                    if k in parent and k not in gv:
                        gv[k] = parent[k]

    def _force_all_gradient_types(node: DictConfig, loss_type: str) -> None:
        """
        Force all existing gradient sections to have the specified loss type.

        :param node: A DictConfig node representing a loss configuration.
        :param loss_type: The loss type to set for all gradient sections.
        """
        g = node.get("gradients")
        if isinstance(g, (dict, DictConfig)):
            for gv in g.values():
                if isinstance(gv, (dict, DictConfig)):
                    gv["type"] = loss_type

    def _new_loss_defaults() -> DictConfig:
        """
        Create a new loss defaults configuration based on CONF_LOSS.

        :return: A DictConfig node representing the loss defaults.
        """
        return OmegaConf.create(CONF_LOSS)

    # Collect per-target energy flags from training_set
    training_confs = conf["training_set"]
    if not isinstance(training_confs, ListConfig):
        training_confs = OmegaConf.create([training_confs])

    per_target_flags: dict[str, dict[str, bool]] = {}
    for tc in training_confs:
        targets = tc.get("targets", {}) or {}
        for tname, opts in targets.items():
            is_energy = (tname == "energy") or (opts.get("quantity") == "energy")

            if is_energy:
                forces_val = opts["forces"]
                stress_val = opts["stress"]
                virial_val = opts["virial"]

                fflag = isinstance(forces_val, (dict, DictConfig))
                sflag = isinstance(stress_val, (dict, DictConfig)) or isinstance(
                    virial_val, (dict, DictConfig)
                )

                if not fflag:
                    assert forces_val is False, (
                        f"'forces' must be a dict or False for energy target '{tname}'"
                    )
                if not (
                    isinstance(stress_val, (dict, DictConfig))
                    or isinstance(virial_val, (dict, DictConfig))
                ):
                    assert (stress_val is False) and (virial_val is False), (
                        "'stress' and 'virial' must be dict or False "
                        f"for energy target '{tname}'"
                    )

            else:
                fflag = False
                sflag = False

            entry = per_target_flags.setdefault(
                tname, {"is_energy": False, "forces": False, "stress": False}
            )
            entry["is_energy"] |= is_energy
            entry["forces"] |= fflag
            entry["stress"] |= sflag

    if not per_target_flags:
        return conf

    # Build defaults per target based on flags (stubs only for forces/stress)
    defaults_map = {}
    for tname, flg in per_target_flags.items():
        base = _new_loss_defaults()
        if flg["is_energy"]:
            base.setdefault("gradients", OmegaConf.create({}))
            if flg["forces"]:
                base["gradients"].setdefault("positions", OmegaConf.create({}))
            if flg["stress"]:
                base["gradients"].setdefault("strain", OmegaConf.create({}))
            # inherit defaults from parent into existing stubs
            _inherit_parent_into_gradients(base)
        defaults_map[tname] = base

    # Normalize user-provided loss into a per-target map
    train_hypers = conf["architecture"]["training"]
    user_loss = train_hypers.get("loss", None)

    string_loss_type = None
    user_loss_map = OmegaConf.create({})

    if isinstance(user_loss, str):
        # apply to all targets later; keep map empty here
        string_loss_type = user_loss
    elif isinstance(user_loss, (dict, DictConfig)):
        # normalize per-target
        for key, val in user_loss.items():
            # keep legacy top-level keys out of per-target map (migrated later with
            # lower precedence)
            if key in ("energy", "forces", "stress", "virial"):
                continue
            if isinstance(val, str):
                user_loss_map[key] = OmegaConf.create({"type": val})
            else:
                user_loss_map[key] = val
            # migrate per-target legacy shorthands to gradients.*
            node = user_loss_map[key]
            if "forces" in node:
                gpos = node.setdefault("gradients", OmegaConf.create({})).setdefault(
                    "positions", OmegaConf.create({})
                )
                fval = node.pop("forces")
                node["gradients"]["positions"] = OmegaConf.merge(
                    {"type": fval} if isinstance(fval, str) else fval, gpos
                )
            for k_legacy in ("stress", "virial"):
                if k_legacy in node:
                    gstr = node.setdefault(
                        "gradients", OmegaConf.create({})
                    ).setdefault("strain", OmegaConf.create({}))
                    sval = node.pop(k_legacy)
                    node["gradients"]["strain"] = OmegaConf.merge(
                        {"type": sval} if isinstance(sval, str) else sval, gstr
                    )
    else:
        # None: nothing per-target from user
        pass

    # Build legacy top-level template (lowest precedence for energy targets)
    legacy_template = OmegaConf.create({})
    if isinstance(user_loss, (dict, DictConfig)):
        # these apply only to energy targets and should not override per-target user
        # settings
        if "energy" in user_loss:
            legacy_template["energy"] = (
                OmegaConf.create({"type": user_loss["energy"]})
                if isinstance(user_loss["energy"], str)
                else user_loss["energy"]
            )
        if "forces" in user_loss:
            legacy_template["forces"] = (
                OmegaConf.create({"type": user_loss["forces"]})
                if isinstance(user_loss["forces"], str)
                else user_loss["forces"]
            )
        if "stress" in user_loss:
            legacy_template["stress"] = (
                OmegaConf.create({"type": user_loss["stress"]})
                if isinstance(user_loss["stress"], str)
                else user_loss["stress"]
            )
        elif "virial" in user_loss:
            legacy_template["stress"] = (
                OmegaConf.create({"type": user_loss["virial"]})
                if isinstance(user_loss["virial"], str)
                else user_loss["virial"]
            )

    # Assemble final per-target loss with correct precedence
    final_loss = OmegaConf.create({})

    # union of targets appearing in training_set and/or user per-target map
    target_names = set(defaults_map.keys()) | set(user_loss_map.keys())

    for tname in target_names:
        # start from defaults (lowest precedence)
        base = defaults_map.get(tname, _new_loss_defaults())

        # layer legacy top-level (low precedence) for energy targets only
        flg = per_target_flags.get(
            tname,
            {"is_energy": False, "forces": False, "stress": False},
        )
        layered = base
        if flg["is_energy"] and isinstance(legacy_template, (dict, DictConfig)):
            # energy root
            if "energy" in legacy_template:
                layered = OmegaConf.merge(layered, legacy_template["energy"])
            # gradients from legacy only if flags say those gradients exist
            if flg["forces"] and "forces" in legacy_template:
                layered.setdefault("gradients", OmegaConf.create({})).setdefault(
                    "positions", OmegaConf.create({})
                )
                layered["gradients"]["positions"] = OmegaConf.merge(
                    layered["gradients"]["positions"], legacy_template["forces"]
                )
            if flg["stress"] and ("stress" in legacy_template):
                layered.setdefault("gradients", OmegaConf.create({})).setdefault(
                    "strain", OmegaConf.create({})
                )
                layered["gradients"]["strain"] = OmegaConf.merge(
                    layered["gradients"]["strain"], legacy_template["stress"]
                )

        # now apply per-target user config (high precedence)
        if tname in user_loss_map:
            layered = OmegaConf.merge(layered, user_loss_map[tname])

        # if loss is passed as a string, force all gradient types to the value defined
        # in the string
        if string_loss_type is not None:
            # ensure parent type is set (it may still be an interpolation)
            layered["type"] = string_loss_type
            _force_all_gradient_types(layered, string_loss_type)

        # inherit missing (type/weight/reduction, and delta-if-present) from parent into
        # existing gradient
        _inherit_parent_into_gradients(layered)

        # finalize huber deltas (single shallow pass)
        _finalize_huber(layered)

        final_loss[tname] = layered

    # write back final loss config
    conf["architecture"]["training"]["loss"] = final_loss
    return conf


def check_units(
    actual_options: Union[DictConfig, ListConfig],
    desired_options: Union[DictConfig, ListConfig],
) -> None:
    """Perform consistency checks between two dataset configs.

    :param actual_options: The dataset options that you want to test.
    :param desired_options: The dataset options ``actual_options`` is tested against.

    :raises ValueError: If the length units are not consistent between
        the system in the dataset options.
    :raises ValueError: If a target is present only in desider_option and
        not in actual_option.
    :raises ValueError: If the unit of a target quantity is not consistent between
        the dataset option.
    """
    if type(actual_options) is DictConfig:
        actual_options = OmegaConf.create([actual_options])
    if type(desired_options) is DictConfig:
        desired_options = OmegaConf.create([desired_options])

    if len(actual_options) != len(desired_options):
        raise ValueError(
            f"Length of actual_options ({len(actual_options)}) and desired_options "
            f"({len(desired_options)}) is different!"
        )

    for actual_options_element, desired_options_element in zip(
        actual_options, desired_options, strict=True
    ):
        actual_length_unit = actual_options_element["systems"]["length_unit"]
        desired_length_unit = desired_options_element["systems"]["length_unit"]

        if actual_length_unit != desired_length_unit:
            raise ValueError(
                "`length_unit`s are inconsistent between one of the dataset options. "
                f"{actual_length_unit!r} != {desired_length_unit!r}."
            )

        for target in actual_options_element["targets"]:
            actual_unit = actual_options_element["targets"][target]["unit"]
            if target in desired_options_element["targets"]:
                desired_unit = desired_options_element["targets"][target]["unit"]
                if actual_unit != desired_unit:
                    raise ValueError(
                        f"Units of target {target!r} are inconsistent between one of "
                        f"the dataset options. {actual_unit!r} != {desired_unit!r}."
                    )
            else:
                raise ValueError(
                    f"Target {target!r} is not present in one of the given dataset "
                    "options."
                )
