import inspect
from typing import Type, TypeVar


HypersType = TypeVar("HypersType")


def init_with_defaults(hypers_cls: Type[HypersType]) -> HypersType:
    """Initialize a TypedDict hypers class with its default values."""
    defaults_dict = {}
    for key in inspect.get_annotations(hypers_cls):
        if key in hypers_cls.__dict__:
            defaults_dict[key] = hypers_cls.__dict__[key]
    return hypers_cls(**defaults_dict)


def get_hypers_annotation(module: type) -> Type[dict]:
    """Get the hypers TypedDict annotation from a module.

    It inspects the signature of the model and gets the annotation
    of the 'hypers' parameter.

    :param module: The module to inspect.

    :return: The TypedDict annotation of the 'hypers' parameter.
    """
    return inspect.signature(module).parameters["hypers"].annotation


# Generic dictionary types to use as annotations in hypers
ScalingWeightsDict = dict
# class ScalingWeightsDict(dict):
#     """Dictionary type for scaling weights."""

#     @classmethod
#     def get_json_schema(cls) -> dict:
#         return {
#             "type": "object",
#             "patternProperties": {
#                 "^.*$": {
#                     "anyOf": [
#                         {
#                             "type": "object",
#                             "propertyNames": {"pattern": "^[0-9]+$"},
#                             "additionalProperties": {"type": "number"},
#                         },
#                         {"type": "number"},
#                     ]
#                 }
#             },
#             "additionalProperties": False,
#         }

CompositionWeightsDict = dict
# class CompositionWeightsDict(dict):
#     """Dictionary type for composition weights."""

#     @classmethod
#     def get_json_schema(cls) -> dict:
#         return {
#             "type": "object",
#             "patternProperties": {
#                 "^.*$": {
#                     "type": "object",
#                     "propertyNames": {"pattern": "^[0-9]+$"},
#                     "additionalProperties": {"type": "number"},
#                 }
#             },
#             "additionalProperties": False,
#         }
