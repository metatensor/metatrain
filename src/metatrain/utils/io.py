import re
import warnings
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union
from urllib.parse import unquote, urlparse
from urllib.request import urlretrieve

import torch
from huggingface_hub import hf_hub_download
from metatomic.torch import check_atomistic_model, load_atomistic_model

from .. import __version__
from .abc import ModelInterface
from .architectures import (
    find_all_architectures,
    get_default_hypers,
    import_architecture,
)
from .wrapper import MetatrainModel


hf_pattern = re.compile(
    r"(?P<endpoint>https://[^/]+)/"
    r"(?P<repo_id>[^/]+/[^/]+)/"
    r"resolve/"
    r"(?P<revision>[^/]+)/"
    r"(?P<filename>.+)"
)


def check_file_extension(
    filename: Union[str, Path], extension: str
) -> Union[str, Path]:
    """Check the file extension of a file name and adds if it is not present.

    If ``filename`` does not end with ``extension`` the ``extension`` is added and a
    warning will be issued.

    :param filename: Name of the file to be checked.
    :param extension: Expected file extension i.e. ``.txt``.
    :return: Checked and probably extended file name.
    """
    path_filename = Path(filename)

    if path_filename.suffix != extension:
        warnings.warn(
            f"The file name should have a '{extension}' file extension. The user "
            f"requested the file with name '{filename}', but it will be saved as "
            f"'{filename}{extension}'.",
            stacklevel=1,
        )
        path_filename = path_filename.parent / (path_filename.name + extension)

    if type(filename) is str:
        return str(path_filename)
    else:
        return path_filename


def is_exported_file(path: str) -> bool:
    """
    Check if a saved model file has been exported to a metatomic ``AtomisticModel``.

    The functions uses :py:func:`metatomic.torch.check_atomistic_model` to verify.

    :param path: model path
    :return: :py:obj:`True` if the ``model`` has been exported, :py:obj:`False`
        otherwise.

    .. seealso::

        :py:func:`metatomic.torch.is_atomistic_model` to verify if an already loaded
        model is exported.
    """
    try:
        check_atomistic_model(str(path))
        return True
    except ValueError:
        return False


def _hf_hub_download_url(
    url: str,
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Wrapper around `hf_hub_download` allowing passing the URL directly.

    Function is in inverse of `hf_hub_url`

    :param url: Full URL to a file in the Hugging Face Hub.
    :param hf_token: HuggingFace API token to download models from HuggingFace
    :param cache_dir: Path to a directory in which downloaded files are cached.
    :return: Path to the downloaded file in the local cache directory.
    """

    match = hf_pattern.match(url)

    if not match:
        raise ValueError(f"URL '{url}' has an invalid format for the Hugging Face Hub.")

    endpoint = match.group("endpoint")
    repo_id = match.group("repo_id")
    revision = unquote(match.group("revision"))
    filename = unquote(match.group("filename"))

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        revision=revision,
        token=hf_token,
        endpoint=endpoint,
    )


def download_model_from_hf(
    repo_id: str,
    filename: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """
    Download a model file from the Hugging Face Hub.

    :param repo_id: The repository ID (e.g., "metatensor/metatrain-test").
    :param filename: The filename within the repository (e.g., "model.ckpt").
    :param revision: The specific revision (branch, tag, or commit hash) to download.
    :param token: Hugging Face API token for private repositories.
    :return: The local path to the downloaded file.
    """
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
    )


def load_model(
    path: Union[str, Path],
    extensions_directory: Optional[Union[str, Path]] = None,
    hf_token: Optional[str] = None,
) -> Any:
    """Load checkpoints and exported models from an URL or a local file for inference.

    Remote models from Hugging Face are downloaded to a local cache directory.

    If an exported model should be loaded and requires compiled extensions, their
    location should be passed using the ``extensions_directory`` parameter.

    After reading a checkpoint, the returned model can be exported with the model's own
    ``export()`` method.

    .. note::

        This function is intended to load models only for **inference** in Python. To
        continue training or to finetune use metatrain's command line interface.

    :param path: local or remote path to a model. For supported URL schemes see
        :py:class:`urllib.request`
    :param extensions_directory: path to a directory containing all extensions required
        by an exported model
    :param hf_token: HuggingFace API token to download (private) models from HuggingFace

    :raises ValueError: if ``path`` is a YAML option file and no model
    :return: the loaded model
    """

    if Path(path).suffix in [".yaml", ".yml"]:
        raise ValueError(
            f"path '{path}' seems to be a YAML option file and not a model"
        )

    path = str(path)
    url = urlparse(path)

    # On Windows, drive letters (e.g. "D") are parsed as single-character URL schemes
    # by urlparse. Only treat as a URL if the scheme is at least 2 characters long.
    if url.scheme and len(url.scheme) > 1:
        if url.netloc == "huggingface.co":
            path = _hf_hub_download_url(url=url.geturl(), hf_token=hf_token)
        else:
            # Avoid caching generic URLs due to lack of a model hash for proper cache
            # invalidation
            path, _ = urlretrieve(url=url.geturl())

    if is_exported_file(path):
        return load_atomistic_model(path, extensions_directory=extensions_directory)
    else:  # model is a checkpoint
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        return model_from_checkpoint(checkpoint, context="export")


def model_from_checkpoint(
    checkpoint: Dict[str, Any],
    context: Literal["restart", "finetune", "export"],
) -> MetatrainModel:
    """
    Load the checkpoint at the given ``path``, and create the corresponding model
    instance. The model architecture is determined from information stored inside the
    checkpoint.

    :param checkpoint: checkpoint dictionary as returned by ``torch.load(path)``.
    :param context: context in which the model is loaded, one of:

        - ``"restart"``: the model is loaded to restart training from a previous
            checkpoint;
        - ``"finetune"``: the model is loaded to finetune a pretrained model;
        - ``"export"``: the model is loaded to export a trained model.

    :return: the loaded model instance.

    """
    if "architecture_name" in checkpoint:
        checkpoint = _ckpt_from_arch_ckpt(checkpoint)

    checkpoint = upgrade_checkpoint(checkpoint)

    if checkpoint["model"]["architecture_name"] == "llpr" and context == "finetune":
        # LLPR returns the wrapped model instead of itself when loading
        # with context="finetune". The wrapped model is already a MetatrainModel.
        # LLPR should not behave like this, but for now we just hack around the hack.
        return arch_model_from_checkpoint(checkpoint["model"], context=context)

    # The normal loading.
    return MetatrainModel(
        model=arch_model_from_checkpoint(checkpoint["model"], context=context),
        additive_models=[
            arch_model_from_checkpoint(additive_model_checkpoint, context=context)
            for additive_model_checkpoint in checkpoint["additive_models"]
        ],
        scaler=arch_model_from_checkpoint(checkpoint["scaler"], context=context)
        if checkpoint["scaler"] is not None
        else None,
        dataset_info=checkpoint["dataset_info"],
    )


# Registry of models that do not belong to an architecture.
_MODEL_REGISTRY = {
    "zbl": "metatrain.utils.additive.zbl.ZBL",
}


def _find_model_class(architecture_name: str) -> Any:
    if architecture_name in _MODEL_REGISTRY:
        module_path, class_name = _MODEL_REGISTRY[architecture_name].rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    else:
        architecture = import_architecture(architecture_name)
        return architecture.__model__


def arch_model_from_checkpoint(
    checkpoint: Dict[str, Any], context: Literal["restart", "finetune", "export"]
) -> ModelInterface:

    checkpoint = upgrade_checkpoint(checkpoint)
    architecture_name = checkpoint["architecture_name"]
    model_cls = _find_model_class(architecture_name)

    return model_cls.load_checkpoint(checkpoint, context=context)


_mtt_model_versions = {
    "pet": 17,
    "llpr": 5,
}


def _ckpt_from_arch_ckpt(checkpoint: dict) -> dict:
    """
    Convert a checkpoint from an architecture to one that can be loaded
    by MetatrainModel.

    This is specially useful for converting checkpoints that were generated
    before the introduction of the MetatrainModel class.
    """
    architecture_name = checkpoint["architecture_name"]
    architecture = import_architecture(architecture_name)
    try:
        # Use the trainer to setup a MetatrainModel
        trainer = trainer_from_checkpoint(checkpoint, context="restart", hypers={})
    except RuntimeError:
        # Create a new trainer.
        default_hypers = get_default_hypers(architecture_name)["training"]
        trainer = architecture.__trainer__(hypers=default_hypers)

    model_cls = architecture.__model__
    if architecture_name in _mtt_model_versions:
        target_version = _mtt_model_versions[architecture_name] - 1
        if checkpoint["model_ckpt_version"] < target_version:
            checkpoint = model_cls.upgrade_checkpoint(
                checkpoint, version=target_version
            )

    model_data = checkpoint["model_data"]
    try:
        mtt_model = trainer.setup(
            model_hypers=model_data.get("hypers", model_data.get("model_hypers")),
            dataset_info=model_data["dataset_info"],
        )
    except:
        mtt_model = MetatrainModel(
            model=arch_model_from_checkpoint(checkpoint, context="export"),
            additive_models=[],
            scaler=None,
            dataset_info=model_data["dataset_info"],
        )

    # Get the new checkpoint skeleton from the MetatrainModel
    new_ckpt = mtt_model.get_checkpoint()

    # Copy all the trainer keys.
    for k in list(checkpoint):
        if k not in new_ckpt["model"]:
            new_ckpt[k] = checkpoint[k]

    # The checkpoint of the model now goes to the "model" key.
    # (here we replace completely the model in new_ckpt, since that one
    # is simply an untrained model)
    new_ckpt["model"] = checkpoint

    # Fill the state dicts of the scaler and additive models by finding
    # them in the state dict of the original checkpoint.
    scaler_state_dict = {}
    additive_models_state_dict = [{} for _ in range(len(new_ckpt["additive_models"]))]

    state_dict = checkpoint["model_state_dict"]
    for k, v in state_dict.items():
        if k.startswith("scaler."):
            scaler_state_dict[k.replace("scaler.", "")] = v
        for i, additive_model_state_dict in enumerate(additive_models_state_dict):
            if k.startswith(f"additive_models.{i}."):
                additive_model_state_dict[k.replace(f"additive_models.{i}.", "")] = v

    if new_ckpt["scaler"] is None:
        if len(scaler_state_dict) > 0:
            raise ValueError(
                "The checkpoint contains a scaler state dict, but the model "
                "does not have a scaler."
            )
    else:
        new_ckpt["scaler"]["model_state_dict"] = scaler_state_dict
        new_ckpt["scaler"]["best_model_state_dict"] = scaler_state_dict

    for i, additive_model_state_dict in enumerate(additive_models_state_dict):
        new_ckpt["additive_models"][i]["model_state_dict"] = additive_model_state_dict
        new_ckpt["additive_models"][i]["best_model_state_dict"] = (
            additive_model_state_dict
        )

    return new_ckpt


def upgrade_checkpoint(checkpoint: dict) -> dict:
    """
    Recursively upgrade a checkpoint.

    It finds all the models in the checkpoint and calls their upgrade function.

    :param checkpoint: The checkpoint to upgrade.
    :raises RuntimeError: if the checkpoint cannot be upgraded to the current
        version of the model.

    :return: The upgraded checkpoint.
    """
    if isinstance(checkpoint, dict) and "architecture_name" in checkpoint:
        # Upgrade this checkpoint.
        architecture_name = checkpoint["architecture_name"]

        known_archs = find_all_architectures() + list(_MODEL_REGISTRY.keys())
        if architecture_name not in known_archs:
            raise ValueError(
                f"Checkpoint architecture '{architecture_name}' not found "
                "in the available architectures. Available architectures are: "
                f"{known_archs}"
            )

        model_cls = _find_model_class(architecture_name)

        model_ckpt_version = checkpoint.get("model_ckpt_version")
        ckpt_before_versioning = model_ckpt_version is None
        if ckpt_before_versioning:
            # assume version 1 and try our best
            model_ckpt_version = 1
            checkpoint["model_ckpt_version"] = model_ckpt_version

        if model_ckpt_version > model_cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to load the model checkpoint for the '{architecture_name}' "
                f"architecture: the checkpoint uses checkpoint format version "
                f"{model_ckpt_version}, but the installed '{architecture_name}' "
                f"architecture only supports up to version "
                f"{model_cls.__checkpoint_version__}. You are using "
                f"metatrain version {__version__}, which is too old to read this "
                "checkpoint. Please upgrade metatrain to a newer version."
            )
        elif model_ckpt_version < model_cls.__checkpoint_version__:
            try:
                if ckpt_before_versioning:
                    warnings.warn(
                        "trying to upgrade an old model checkpoint with unknown "
                        "version, this might fail and require manual modifications",
                        stacklevel=1,
                    )

                checkpoint = model_cls.upgrade_checkpoint(checkpoint)
            except Exception as e:
                raise
                raise RuntimeError(
                    f"Unable to load the model checkpoint for "
                    f"the '{architecture_name}' architecture: the checkpoint is using "
                    f"version {model_ckpt_version}, while the current version is "
                    f"{model_cls.__checkpoint_version__}; and trying to "
                    "upgrade the checkpoint failed."
                ) from e
        checkpoint = model_cls.upgrade_checkpoint(checkpoint)

    # Continue to recursively search for models in the checkpoint and upgrade them.
    if isinstance(checkpoint, dict):
        for k, v in checkpoint.items():
            checkpoint[k] = upgrade_checkpoint(v)
    elif isinstance(checkpoint, list):
        for i, v in enumerate(checkpoint):
            checkpoint[i] = upgrade_checkpoint(v)

    return checkpoint


def trainer_from_checkpoint(
    checkpoint: Dict[str, Any],
    context: Literal["restart", "finetune", "export"],
    hypers: Dict[str, Any],
) -> Any:
    """
    Load the checkpoint at the given ``path``, and create the corresponding trainer
    instance. The architecture is determined from information stored inside the
    checkpoint.

    :param checkpoint: checkpoint dictionary as returned by ``torch.load(path)``.
    :param context: context in which the trainer is loaded, one of:

        - ``"restart"``: the trainer is loaded to restart training from a previous
            checkpoint;
        - ``"finetune"``: the trainer is loaded to finetune a pretrained model;
        - ``"export"``: the trainer is loaded to export a trained model.

    :param hypers: hyperparameters to be used by the trainer. Required if
        ``context="finetune"``, ignored otherwise.
    :return: the loaded trainer instance.

    """
    if "architecture_name" in checkpoint:
        # This is an old checkpoint
        architecture_name = checkpoint["architecture_name"]
    else:
        architecture_name = checkpoint["model"]["architecture_name"]

    if architecture_name not in find_all_architectures():
        raise ValueError(
            f"Checkpoint architecture '{architecture_name}' not found "
            "in the available architectures. Available architectures are: "
            f"{find_all_architectures()}"
        )
    architecture = import_architecture(architecture_name)

    trainer_ckpt_version = checkpoint.get("trainer_ckpt_version")

    ckpt_before_versioning = trainer_ckpt_version is None
    if ckpt_before_versioning:
        # assume version 1 and try our best
        trainer_ckpt_version = 1
        checkpoint["trainer_ckpt_version"] = trainer_ckpt_version

    base_msg = (
        f"Unable to load the trainer checkpoint for the '{architecture_name}' "
        "architecture:"
    )
    if trainer_ckpt_version > architecture.__trainer__.__checkpoint_version__:
        raise RuntimeError(
            f"{base_msg} the checkpoint uses checkpoint format version "
            f"{trainer_ckpt_version}, but the installed '{architecture_name}' "
            f"architecture only supports up to version "
            f"{architecture.__trainer__.__checkpoint_version__}. You are using "
            f"metatrain version {__version__}, which is too old to read this "
            "checkpoint. Please upgrade metatrain to a newer version."
        )
    elif trainer_ckpt_version < architecture.__trainer__.__checkpoint_version__:
        try:
            if ckpt_before_versioning:
                warnings.warn(
                    "trying to upgrade an old trainer checkpoint with unknown "
                    "version, this might fail and require manual modifications",
                    stacklevel=1,
                )

            checkpoint = architecture.__trainer__.upgrade_checkpoint(checkpoint)
        except Exception as e:
            raise RuntimeError(
                f"{base_msg} the checkpoint uses checkpoint format version "
                f"{trainer_ckpt_version}, while the current version is "
                f"{architecture.__trainer__.__checkpoint_version__}; and trying to "
                "upgrade the checkpoint failed."
            ) from e

    return architecture.__trainer__.load_checkpoint(
        checkpoint, context=context, hypers=hypers
    )
