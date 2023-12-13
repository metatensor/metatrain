import argparse


def _add_export_model_parser(subparser: argparse._SubParsersAction) -> None:
    parser = subparser.add_parser(
        "export",
        description=export_model.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(callable="export_model")


def export_model():
    """export a model"""
    print("Run exort...")
